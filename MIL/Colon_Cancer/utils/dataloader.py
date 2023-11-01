"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch

class Colon_Dataset(torch.utils.data.Dataset):
    def __init__(self, curr_fold):
        'Initialization'
        self.train = curr_fold
        self.labels_train= []
        self.bags_train = []

        for tup in self.train:
            self.labels_train.append(max(tup[1]))
            self.bags_train.append(np.transpose(tup[0], (0, 3, 1, 2)))
        


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.bags_train)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = torch.tensor(np.array(self.bags_train[index]))
        Y = torch.tensor([self.labels_train[index]])
        return X, Y
