'''Linear.'''
import torch.nn as nn
import torch.nn.functional as F
import torch

def logsigmoid(x):
    return -torch.nn.functional.softplus(-x)

class LinearOut1(nn.Module):
    def __init__(self,input_dim=784, out_dim = 1):
        super(LinearOut1, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.lin1 = nn.Linear(input_dim, out_dim)
        self.logsigmoid2 = logsigmoid
    def forward(self, x):
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.logsigmoid2(x).clamp(max=-torch.finfo().eps)
        return x