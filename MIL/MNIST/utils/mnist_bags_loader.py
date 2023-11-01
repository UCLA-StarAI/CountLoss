"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

"""Pytorch Dataset object that loads perfectly balanced MNIST dataset in bag form."""

import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=1, num_bag=1000, seed=7, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.seed = seed
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000


        if self.train:
            self.train_bags_list, self.train_labels_list = self._form_bags()
            self.train_instance = []
            self.train_instance_labels = []
            for bag in self.train_bags_list:
                for inst in bag:
                    self.train_instance.append(inst)
            for bag_inst in self.train_instance_labels:
                for inst_label in bag_inst:
                    self.train_instance_labels.append(inst_label)

        else:
            self.test_bags_list, self.test_labels_list = self._form_bags()
            self.test_instance = []
            self.test_instance_labels = []
            for bag in self.test_bags_list:
                for inst in bag.tolist():
                    self.test_instance.append(inst)
            for bag_inst in self.test_labels_list:
                for inst_label in bag_inst.tolist():
                    self.test_instance_labels.append(int(inst_label))
    def return_data(self):
        if self.train:
            return self.train_instance, self.train_instance_labels
        else:
            return torch.tensor(self.test_instance), torch.tensor(self.test_instance_labels)
    def _form_bags(self):
        if self.train:
            train_loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))])),
                                                 batch_size=self.num_in_train,
                                                 shuffle=False)

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in train_loader:
                numbers = batch_data[0]
                labels = batch_data[1]
            #import pdb; pdb.set_trace()
            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
                labels_in_bag = labels[indices]
                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_train, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1
                    index_list2 = torch.LongTensor([]);
                    for index_tens in index_list:
                        index_list2 = torch.cat((index_list2, index_tens), 0)
                    
                    #index_list2 = np.array(index_list2)
                    labels_in_bag = labels[index_list2]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list2])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        else:
            test_loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                               train=False,
                                                               download=True,
                                                               transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))])),
                                                batch_size=self.num_in_test,
                                                shuffle=False)

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in test_loader:
                numbers = batch_data[0]
                labels = batch_data[1]

            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))
                labels_in_bag = labels[indices]

                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_test, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list2 = torch.LongTensor([]);
                    for index_tens in index_list:
                        index_list2 = torch.cat((index_list2, index_tens), 0)
                    labels_in_bag = labels[index_list2]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list2])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":
    to_pil = transforms.Compose([transforms.ToPILImage()])

    kwargs = {}
    batch_size = 1

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=50,
                                                   var_bag_length=10,
                                                   num_bag=100,
                                                   seed=98,
                                                   train=True),
                                         batch_size=batch_size,
                                         shuffle=False, **kwargs)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=50,
                                                  var_bag_length=10,
                                                  num_bag=10,
                                                  seed=98,
                                                  train=False),
                                        batch_size=batch_size,
                                        shuffle=False, **kwargs)

    len_bag_list = []
    mnist_bags_train = 0
    for batch_idx, data in enumerate(train_loader):
        plot_data = data[0].squeeze(0)
        len_bag_list.append(int(plot_data.size()[0]))
        # plot_data = data[0].squeeze(0)
        # num_instances = int(plot_data.size()[0])
        # print(data[1][0])
        # for i in range(num_instances):
        #     plt.subplot(num_instances, 1, i + 1)
        #     to_pil(plot_data[i, :, :, :]).show()
        # plt.show()
        if data[1][0][0] == 1:
            mnist_bags_train += 1
    print('number of bags with 9(s): ', mnist_bags_train)
    print('total number of bags', len(train_loader))
    print(np.mean(len_bag_list), np.min(len_bag_list), np.max(len_bag_list))

    len_bag_list = []
    mnist_bags_test = 0
    for batch_idx, data in enumerate(test_loader):
        plot_data = data[0].squeeze(0)
        len_bag_list.append(int(plot_data.size()[0]))
        if data[1][0][0] == 1:
            mnist_bags_test += 1
    print('number of bags with 9(s): ', mnist_bags_test)
    print('total number of bags', len(test_loader))
    print(np.mean(len_bag_list), np.min(len_bag_list), np.max(len_bag_list))
