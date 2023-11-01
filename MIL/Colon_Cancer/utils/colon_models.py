import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

def logsigmoid(x):
    return -torch.nn.functional.softplus(-x)

class Colon_Models(nn.Module):
    def __init__(self):
        super(Colon_Models, self).__init__()
        self.conv1 = nn.Conv2d(3, 36, kernel_size=4, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(36, 48, kernel_size=3, stride=1)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(1200, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        self.log_sigmoid2 = logsigmoid



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        output = self.log_sigmoid2(x).clamp(max=-torch.finfo().eps)
        return output
