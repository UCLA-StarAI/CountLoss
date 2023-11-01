import torch
from torchvision import transforms


class Dataset_Bags(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  
  def __init__(self, bags, proportion_label, count_label):
        'Initialization'
        self.proportion_label = proportion_label
        self.bags = bags
        self.count_label = count_label

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.bags)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.bags[index]
        y = self.proportion_label[index]
        z = self.count_label[index]
        return X, y, z
  def _return_all(self):
       return self.bags, self.proportion_label, self.count_label