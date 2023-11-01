import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import *
from models.bert import initialize_bert_based_model
from models.linear_out1 import LinearOut1
from models.MLP import MLP1, MLP2
from models.ResNet1 import ResNet18_1_classes
from numpy.core.numeric import False_


def logsigmoid(x):
    return -torch.nn.functional.softplus(-x)


def get_model(model_type, input_dim=None): 
    if model_type == 'FCN':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, 5000, bias=True),
            nn.ReLU(),
            nn.Linear(5000, 5000, bias=True),
            nn.ReLU(),
            nn.Linear(5000, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 2, bias=True)
        )
        return net
    elif model_type == 'UCI_FCN':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 2, bias=True)
        )
        return net
    elif model_type == 'linear':
        net = nn.Sequential(nn.Flatten(),
            nn.Linear(input_dim, 2, bias=True),
        )
        return net
    elif model_type == 'ResNet':
        net = ResNet18(num_classes=2)
        return net
    elif model_type == 'LeNet':
        net = LeNet(num_classes=2)
        return net
    elif model_type == 'AllConv': 
        net = AllConv()
        return net
    elif model_type == "DistilBert":
        net = initialize_bert_based_model("distilbert-base-uncased", num_classes=2)
        return net 
    elif model_type == "linear_out1":
        net = LinearOut1(input_dim=input_dim)
        return net
    elif model_type == "ResNet_out1":
        net = ResNet18_1_classes(num_classes=1)
        return net
    elif model_type == "AllConv_out1":
        net = AllConv2()
        return net
    elif model_type == "MLP1":
        net = MLP1(input_dim=input_dim)
        return net
    elif model_type == "MLP2":
        net = MLP2(input_dim=input_dim)
        return net
    else:
        print("Model type must be one of FCN | CNN | linear ... ")
        sys.exit(0)


def train_penultimate(net, model_type): 
    if model_type == 'FCN': 
        for param in net.parameters(): 
            param.requires_grad = False

        for param in net.module[-1].parameters():
            param.requires_grad = True

    
    return net
