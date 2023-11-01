import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

def logsigmoid(x):
    return -torch.nn.functional.softplus(-x)

class MNIST_PAPER_MODEL(torch.nn.Module):
    def __init__(self):
        super(MNIST_PAPER_MODEL, self).__init__()
        #add more layers
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.log_sigmoid2 = logsigmoid


        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
        )
        self.fc2 = torch.nn.Linear(500, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.feature_extractor_part1(x)
        x = x.view(-1, 50*4*4)
        x = self.feature_extractor_part2(x)
        x = self.log_sigmoid2(x).clamp(max=-torch.finfo().eps)
        return x
    
class MNIST_PAPER_MODEL_SIG(torch.nn.Module):
    def __init__(self):
        super(MNIST_PAPER_MODEL_SIG, self).__init__()
        #add more layers
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.sig = nn.Sigmoid()


        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
        )
        self.fc2 = torch.nn.Linear(500, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.feature_extractor_part1(x)
        x = x.view(-1, 50*4*4)
        x = self.feature_extractor_part2(x)
        x = self.sig(x)
        return x