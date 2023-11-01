import torch

def logsigmoid(x):
    return -torch.nn.functional.softplus(-x)

class LR(torch.nn.Module):
    def __init__(self, input_dim):
        super(LR, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        x = self.layer1(x)
        return torch.sigmoid(x)

class Adult_Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(Adult_Model, self).__init__()
        #add more layers
        self.layer1 = torch.nn.Linear(input_dim, 2048)
        self.layer2 = torch.nn.Linear(2048, 64)
        self.layer3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.log_sigmoid = logsigmoid
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.log_sigmoid(x).clamp(max=-torch.finfo().eps)
        return x

class Magic_Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(Magic_Model, self).__init__()
        #add more layers
        self.layer1 = torch.nn.Linear(input_dim, 2048)
        self.layer2 = torch.nn.Linear(2048, 1)
        self.relu = torch.nn.ReLU()
        self.log_sigmoid = logsigmoid
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.log_sigmoid(x).clamp(max=-torch.finfo().eps)
        return x