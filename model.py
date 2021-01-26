# coding: UTF-8
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from torchvision.datasets import CIFAR10

import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# from zero_out import zero_out
import numpy as np
import math


# from utils import norm_col_init, choose_action, gaussian_policy, weights_init_mlp, weights_init

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(14 * 14 * 32, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(1,1,28,28)
        #x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = x.view(-1, 14 * 14 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


