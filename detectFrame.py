import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import os

import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import pickle as pk


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)


class capCapture(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def  classes(self):
        return self.data.classes
    

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# def EncodeFiles():
#     file = open('EncodeFile.p', 'rb')
#     encodeListKnownWithIds = pk.load(file)
#     file.close()
#     return encodeListKnownWithIds

# path = 'dataset'
dataset = capCapture(data_dir= './dataset', transform=transform)
# print('dataset length: ',len(dataset))
# print(dataset[0])

# for img , id in dataset:
#     print(img.shape)

# print(EncodeFiles())

# encodeListKnown, EmployeeIds = EncodeFiles()
# for img in encodeListKnown:
#     # img.shape
#     plt.show(img.shape)


dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for img , labels in dataloader:
#     print(img.shape)


class faceClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(faceClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280 
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
        
    
    
model = faceClassifier(num_classes=53)
print(str(model)[:500])


