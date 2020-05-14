"""
Model for transfer learning from CheXNet by training
only the output layer (last fully-connected one).
We are using here the "freezing" approach.
"""
# PyTorch imports
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms, utils

# Image imports
from skimage import io, transform
from PIL import Image

# General imports
import os
import time
from shutil import copyfile
from shutil import rmtree

import pandas as pd
import numpy as np
import csv

import cxr_dataset as CXR
import eval_model as E

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def load_checkpoint():
    #####
    # The pre-trained model checkpoint from 'reproduce-chexnet' contains:
    # state = {
    #     'model': model,
    #     'best_loss': best_loss,
    #     'epoch': epoch,
    #     'rng_state': torch.get_rng_state(),
    #     'LR': LR
    # }
    #####
    
    # Locate and load CheXNet model
    pretrained_checkpoint = '../pretrained_chexnet/checkpoint'
    chexnet_checkpoint = torch.load(pretrained_checkpoint, map_location=torch.device('cpu'))
    model_tl = models.densenet121(pretrained=False)
    chexnet_model = chexnet_checkpoint['model']
    model_tl.load_state_dict(chexnet_model.state_dict())
    # epoch = chexnet_checkpoint['epoch']
    # loss = chexnet_checkpoint['loss']
    # LR = chexnet_checkpoint['LR']
    del chexnet_checkpoint
    return model_tl


def checkpoint(model, best_loss, epoch, LR):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
        optimizer: pytorch optimizer to be saved
    Returns:
        None
    """
    state = {
        'model': model.state_dict(),
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR,
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, 'results/tl_pretraining_checkpoint')


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0] * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY):
    """
    Trains model to COVID-19 dataset.

    Args:
        PATH_TO_IMAGES: path to COVID-19 image data collection
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = 20
    BATCH_SIZE = 30

    try:
        rmtree('results/')
    except BaseException:
        pass
    os.makedirs("results/")

    # ImageNet parameters for normalization
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    # Binary classifier
    N_LABELS = 2

    # load labels
    df = pd.read_csv("covid19_labels.csv", index_col=0)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(PATH_TO_IMAGES, x), 
                                              data_transforms[x]) 
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, 
                                                  shuffle=True, num_workers=4) 
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create train/val dataloaders
    # transformed_datasets = {}
    # transformed_datasets['train'] = CXR.CXRDataset(
    #     path_to_images=PATH_TO_IMAGES,
    #     fold='train',
    #     transform=data_transforms['train'])
    # transformed_datasets['val'] = CXR.CXRDataset(
    #     path_to_images=PATH_TO_IMAGES,
    #     fold='val',
    #     transform=data_transforms['val'])

    # dataloaders = {}
    # dataloaders['train'] = torch.utils.data.DataLoader(
    #     transformed_datasets['train'],
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=8)
    # dataloaders['val'] = torch.utils.data.DataLoader(
    #     transformed_datasets['val'],
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=8)

    # Verify if GPU is available
    # if not use_gpu:
    #     raise ValueError("Error, requires GPU")
    
    num_ftrs = model_tl.fc.in_features
    # Size of each output sample.
    model_tl.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Softmax())
    
    # Define Loss Function (Binary Cross-Entropy Loss)
    criterion = nn.BCELoss()
    # Define optimizer for the new model
    # With Adam Optimizer
    optimizer = optim.Adam(model_tl.parameters(), lr=0.001)
    # With SGD Optimizer
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Observe that all parameters are being optimized
    criterion = nn.CrossEntropyLoss()
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    if use_gpu:
        model_tl = model_tl.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES)

    return preds, aucs
