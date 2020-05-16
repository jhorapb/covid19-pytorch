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
from torchvision import (
    datasets, 
    models, 
    transforms, 
    utils
)

# Image imports
# from skimage import io, transform
# from PIL import Image

# General imports
import os
import re
import time
from shutil import copyfile
from shutil import rmtree
from pathlib import Path

# import pandas as pd
import numpy as np
import csv

# import covid_dataset as COVID_XR
# import eval_model as E


RESULTS_PATH = '../results/'
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
print("Device: " + str(device))
gpu_count = torch.cuda.device_count()
print("Available GPU count: " + str(gpu_count))


def load_checkpoint(PATH_CHECKPOINT, mode='state_dict'):
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
    
    # Define new base model
    model_tl = models.densenet121(pretrained=False)
    model_dict = model_tl.state_dict()
    
    # Locate checkpoint
    chexnet_checkpoint = torch.load(PATH_CHECKPOINT, map_location=torch.device('cpu'))
    if mode == 'state_dict':
        # Load pretrained CheXNet model (mode state_dict)
        state_dict_chexnet = chexnet_checkpoint['state_dict']
        # model_tl = torch.nn.DataParallel(model_tl)
    else:
        # Load pretrained CheXNet model (mode full_model)
        chexnet_model = chexnet_checkpoint['model']
        state_dict_chexnet = chexnet_model.state_dict()
    
    # 1. Filter out unnecessary keys
    state_dict_chexnet = {k: v for k, v in state_dict_chexnet.items() 
                          if k in model_dict}
    # 2. Overwrite entries in the existing state dict
    # model_tl.update(state_dict_chexnet)
    # 3. Load the new state dict
    model_tl.load_state_dict(model_dict)    
    
    # epoch = chexnet_checkpoint['epoch']
    # loss = chexnet_checkpoint['loss']
    # LR = chexnet_checkpoint['LR']
    
    # Freeze the parameters for feature extraction
    for parameter in model_tl.parameters():
        parameter.requires_grad = False

    # If model is used for inference, then evaluate it
    # model.eval()
    
    del chexnet_checkpoint
    return model_tl


def save_checkpoint(model, best_loss, epoch, LR):
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

    torch.save(state, RESULTS_PATH + 'tl_pretraining_checkpoint')


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
    Fine tunes torchvision model to COVID-19 CXR data.

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
    best_acc = 0.0
    best_epoch = -1
    last_train_loss = -1

    # Iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 17)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                # model.train(False)
                model.eval()

            running_loss = 0.0
            
            total_done = 0
            
            # Iterate over dataset (train/val)
            for inputs, labels in dataloaders[phase]:
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device)).float()
                outputs = model(inputs)

                optimizer.zero_grad()
                # Compute loss
                loss = criterion(outputs, labels)
                # Backward pass: compute gradient and update 
                # parameters in training 
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # loss.item()
                running_loss += loss.data[0] * batch_size
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print('{} epoch {}=> Loss: {:.4f} | Acc: {:.4f} | Data size: {}'.format(
                phase, epoch, epoch_loss, epoch_acc, dataset_sizes[phase]))

            if phase == 'val':
                # # Decay learning rate if validation loss plateaus in this epoch
                # if epoch_loss > best_loss:
                #     decayed_LR = LR / 10
                #     print('Decay Loss from {} to {} \
                #             as not seeing improvement in val loss'.format(
                #                 str(LR), str(decayed_LR))
                #             )
                    # LR = decayed_LR
                    # # Create new optimizer with lower learning rate
                    # optimizer = optim.Adam(
                    #     filter(
                    #         lambda p: p.requires_grad, 
                    #         model_tl.parameters()), 
                    #     lr=LR, betas=(0.9, 0.999))
                #     print("Created new optimizer with LR " + str(LR))
                
                # Checkpoint model if has best val loss yet
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_epoch = epoch
                    save_checkpoint(model, best_loss, epoch, LR)

                # Log training and validation loss over each epoch
                with open(RESULTS_PATH + '/log_train', 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # Apply early stopping if there is no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {:4f}'.format(best_acc))

    # load best model weights to return
    checkpoint_best = torch.load(RESULTS_PATH + 'checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch


def perform_tl_cnn(PATH_TO_IMAGES, CHEXNET_CHECKPOINT, checkpoint_type, 
                   LR, WEIGHT_DECAY):
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
    # Since the COVID-19 dataset at the moment is considerably small, 
    # it makes sense to use Batch Gradient Descent (all the samples 
    # being used to update the model parameters)
    minibatch_gd = False
    BATCH_SIZE = 375 if not minibatch_gd else 30

    # Create path to save model results
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    
    # try:
    #     rmtree(RESULTS_PATH)
    # except BaseException:
    #     pass
    # os.makedirs(RESULTS_PATH)

    # ImageNet parameters for normalization
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    # Binary classifier
    N_LABELS = 1
    # Multi-class classifier
    # N_LABELS = 2

    # load labels
    # df = pd.read_csv("covid19_labels.csv", index_col=0)

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
    
    # Option num. workers 8
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                                                  shuffle=True, num_workers=4) 
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print('CLASS NAMES\n:', class_names, '\n')
    # Verify if GPU is available
    # if not use_gpu:
    #     raise ValueError("Error, requires GPU")
    # if use_gpu:
    #     model_tl = model_tl.cuda()
    
    # Load pre-trained CheXNet model
    model_tl = load_checkpoint(CHEXNET_CHECKPOINT, mode=checkpoint_type)
    print('Pre-trained Model:\n', model_tl)
    num_ftrs = model_tl.classifier.in_features
    # Size of each output sample.
    model_tl.classifier = nn.Linear(num_ftrs, N_LABELS)
    # If multiple-class classifier were used, a Sequential
    # container would be necessary. 
    # E.g., nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Softmax())
    
    # Define Loss Function (Binary Cross-Entropy Loss)
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    # Define optimizer for the new model
    # With Adam Optimizer
    optimizer = optim.Adam(model_tl.parameters(), lr=LR, betas=(0.9, 0.999))
    # With SGD Optimizer
    # Observe that all parameters are being optimized
    # optimizer = optim.SGD(model_tl.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs (when using SGD optimizer)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Train COVID model
    model, best_epoch = train_model(model_tl, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY)

    # get preds and AUCs on test fold
    # preds, aucs = E.make_pred_multilabel(
    #     data_transforms, model, PATH_TO_IMAGES)
    
    return model


if __name__ == "__main__":
    binary_classifier = True
    if binary_classifier:
        PATH_TO_IMAGES = "../images/cleaned_up/binary_classifier"
    else:
        PATH_TO_IMAGES = "../images/cleaned_up/multiclass_classifier"
    
    checkpoint_type = 'full_model'
    CHEXNET_CHECKPOINT = '../pretrained_chexnet/checkpoint'
    # checkpoint_type = 'state_dict'
    # CHEXNET_CHECKPOINT = '../pretrained_chexnet/m-25012018-123527.pth.tar'
    
    # Hyperparams for Adam Optimizer: LR=0.001, betas=(0.9, 0.999)
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    best_model = perform_tl_cnn(PATH_TO_IMAGES, CHEXNET_CHECKPOINT, checkpoint_type, 
                                LEARNING_RATE, WEIGHT_DECAY)
    print(best_model)