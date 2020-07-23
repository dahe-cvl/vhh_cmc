import torch
import torchvision.transforms as transforms
from torchvision import models
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
        model = models.resnet50(pretrained=True)

        for params in model.parameters():
            params.requires_grad = False

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
            input_size=2048,  # 2560, # 15360,   #100224,57344 8192
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


def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=128, img_y=128, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=3):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self.frontend(1, 32)
        self.conv_layer2 = self.block(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv_layer3 = self.block(64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv_layer4 = self.block(128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.conv_layer5 = self.block(256, kernel_size=(3, 3, 3), stride=(1, 1, 1), pool=False)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 7, 7))


        #self.conv_layer1 = self._conv_layer_set_single(1, 64, kernel_size=(7, 7, 7), stride=2)
        #self.conv_layer2 = self._conv_layer_set_double(64, 128, kernel_size=(3, 3, 3))
        #self.conv_layer3 = self._conv_layer_set_double(128, 256, kernel_size=(3, 3, 3))
        #self.conv_layer4 = self._conv_layer_set_double(256, 512, kernel_size=(3, 3, 3))
        #self.conv_layer5 = self._conv_layer_set_double(512, 512, kernel_size=(3, 3, 3))
        #self.conv_layer6 = self._conv_layer_set(256, 512)

        #self.avgpool = nn.AdaptiveAvgPool3d((1, 7, 7))

        self.fc1 = nn.Linear(25088, 512)
        self.batch = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 3)
        #self.softmax = nn.Softmax(dim=1)

    def _conv_layer_set_single(self, in_c, out_c, kernel_size=(3, 3, 3), stride=1):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            #nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), padding=1),
            #nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def frontend(self, nin, nout):
        return nn.Sequential(
            nn.Conv3d(nin, nout, kernel_size=(3, 7, 7), padding=(1, 3, 3), stride=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

    def block(self, nin, nout=None, pool=True, kernel_size=(3, 3, 3), stride=(1, 2, 2)):
        nout = nout if nout else nin * 2

        return nn.Sequential(
            nn.Conv3d(nin, nout, kernel_size=kernel_size, stride=stride, padding=(1, 1, 1)),
            nn.Conv3d(nout, nout, kernel_size=kernel_size, stride=stride, padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) if pool else nn.Identity()
        )

    def _conv_layer_set_double(self, in_c, out_c, kernel_size=(3, 3, 3), stride=1):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, in_c, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm3d(in_c),
            nn.ReLU(),
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        #print(x.size())
        out = self.conv_layer1(x)
        #print(out.size())
        out = self.conv_layer2(out)
        #print(out.size())
        out = self.conv_layer3(out)
        #print(out.size())
        out = self.conv_layer4(out)
        #print(out.size())
        out = self.conv_layer5(out)
        #print(out.size())

        out = self.avgpool(out)

        out = out.view(out.size(0), -1)


        #print(out.size())

        #exit()

        out = self.fc1(out)
        out = self.batch(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        #out = self.softmax(out)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=400, last_fc=True):
        self.last_fc = last_fc

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model