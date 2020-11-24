import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from PIL import Image
import torch.nn as nn  # Add on classifier
import os
import cv2

class CNN(nn.Module):
    """CNN."""

    def __init__(self, n_classes=3):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer_frontend = nn.Sequential(
            # frontend
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        """Perform forward."""
        #print(x.size())

        # frontend
        x = self.conv_layer_frontend(x)
        #print(x.size())

        # conv layers
        x = self.conv_layer(x)
        #print(x.size())

        # flatten
        x = x.view(x.size(0), -1)
        #print(x.size())

        # fc layer
        #x = self.fc_layer(x)

        return x


class ResnetCnn(nn.Module):
    """CNN."""

    def __init__(self, n_classes, pre_trained_path=None):
        """CNN Builder."""
        super(ResnetCnn, self).__init__()
        model = models.resnet18(pretrained=True)

        for params in model.parameters():
            params.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

        if(pre_trained_path != None):
            model_dict_state = torch.load(pre_trained_path + "/best_model.pth")
            model.load_state_dict(model_dict_state['net'])

        self.conv_features = nn.Sequential(*list(model.children())[:-2])
        self.avg_features = nn.Sequential(*list(model.children())[:-1])

        print(model)
        #exit()

    def forward(self, x):
        """Perform forward."""
        # print(x.size())
        # conv layers
        x = self.avg_features(x)
        # x = self.pre_model_features(x)
        #print(x.size())
        #exit()

        # flatten
        #x = x.view(x.size(0), -1)
        #print(x.size())

        # fc layer
        # x = self.fc_layer(x)

        return x

class CnnLstm(nn.Module):
    def __init__(self):
        super(CnnLstm, self).__init__()
        self.cnn = ResnetCnn(3)
        self.rnn = nn.LSTM(
            input_size=512,  # 2560, # 15360,   #100224,57344 8192
            hidden_size=16,
            num_layers=1,
            dropout=0.2,
            batch_first=True,
            bidirectional=False
        )
        self.linear = nn.Linear(16, 3)

    def forward(self, x):
        #print("-------------")
        #print(x.size())
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        #print(c_in.size())
        c_out = self.cnn(c_in)
        #print(c_out.size())
        r_in = c_out.view(batch_size, timesteps, -1)
        #print(r_in.size())
        #r_in = r_in.view(batch_size, -1, timesteps)
        #print(r_in.size())

        r_out, (h_n, h_c) = self.rnn(r_in)
        #print(r_out.size())
        #exit()
        r_out2 = self.linear(r_out[:, -1, :])
        out = r_out2
        #print(r_out2.size())
        #out = F.log_softmax(r_out2, dim=1)
        #print(out.size())
        #exit()
        return out


# Create simple CNN Model
class Cnn3dModel(nn.Module):
    def __init__(self, num_classes=3):
        super(Cnn3dModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(3, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.conv_layer3 = self._conv_layer_set(64, 128)
        self.conv_layer4 = self._conv_layer_set(128, 256)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

        self.log_soft = nn.LogSoftmax(dim=1) 
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        DEBUG_FLAG = False
        if(DEBUG_FLAG == True): print(x.size())

        # feature extractor
        x = self.conv_layer1(x)
        if(DEBUG_FLAG == True): print(x.size())
        x = self.conv_layer2(x)
        if(DEBUG_FLAG == True): print(x.size())
        x = self.conv_layer3(x)
        if(DEBUG_FLAG == True): print(x.size())
        x = self.conv_layer4(x)
        if(DEBUG_FLAG == True): print(x.size())

        # flatten features
        x = x.view(x.size(0), -1)
        if(DEBUG_FLAG == True): print(x.size())

        # fully connected layers
        x = self.fc1(x)
        if(DEBUG_FLAG == True): print(x.size())
        x = self.relu(x)
        x = self.batch(x)
        x = self.drop(x)

        if(DEBUG_FLAG == True): print(x.size())
        x = self.fc2(x)
        x = self.log_soft(x)

        if(DEBUG_FLAG == True): print(x.size())
        if(DEBUG_FLAG == True): print(x.size())
        return x
