"""
Model for transfer learning from CheXNet by training
only the output layer (last fully-connected one).
We are using here the "freezing" approach.
"""
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

pretrained_checkpoint = '../pretrained_chexnet/checkpoint'
chexnet_checkpoint = torch.load(pretrained_checkpoint, map_location=torch.device('cpu'))

model = models.densenet121(pretrained=False)
chexnet_model = chexnet_checkpoint['model']
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
model.load_state_dict(chexnet_model.state_dict())
# optimizer.load_state_dict(chexnet_checkpoint['optimizer_state_dict'])
epoch = chexnet_checkpoint['epoch']
loss = chexnet_checkpoint['loss']
del chexnet_checkpoint

model.classifier
# model.eval()
# - or -
# model.train()