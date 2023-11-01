'''MLP'''
import torch.nn as nn
import torch.nn.functional as F
import torch

def logsigmoid(x):
    return -torch.nn.functional.softplus(-x)

class MLP1(nn.Module):
    def __init__(self,input_dim=784, out_dim = 1):
        super(MLP1, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.lin1 = nn.Linear(input_dim, 5000, bias=True)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(5000, 5000, bias=True)
        self.lin3 = nn.Linear(5000, 50, bias=True)
        self.lin4 = nn.Linear(50, 1, bias=True)
        self.logsigmoid2 = logsigmoid
    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.logsigmoid2(x).clamp(max=-torch.finfo().eps)
        return x
class MLP2(nn.Module):
    def __init__(self,input_dim=784, out_dim = 1):
        super(MLP2, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.lin1 = nn.Linear(input_dim, 512, bias=True)
        self.lin2 = nn.Linear(512, 512, bias=True)
        self.lin3 = nn.Linear(512, 1, bias=True)
        self.relu = nn.ReLU()
        self.logsigmoid2 = logsigmoid
    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.logsigmoid2(x).clamp(max=-torch.finfo().eps)
        return x