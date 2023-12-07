import os

os.environ["OMP_NUM_THREADS"] = "20" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "20" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "20" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "20" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "20" # export NUMEXPR_NUM_THREADS=1

import argparse
import random

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from utils.log_space_algo import *
from utils.logspace import *
from utils.mnist_bags_loader import *
from utils.mnist_models import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_acc_auc_instance(model, X, Y):
    # calculate instance acc and auc
    model.eval()
    Y = Y.to(device)
    X = X.to(device)
    X = torch.squeeze(torch.exp(model(X))).round().detach().numpy()
    acc_score = accuracy_score(Y, X)
    auc_score = roc_auc_score(Y, X)
    return acc_score, auc_score

def calculate_auc_bag_probalistic(model, test_loader):
    # calculate bag level metrics
    model.eval()
    pred_vals = []
    true_vals = []
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        data, bag_label = data.to(device), bag_label.to(device)
        bag_label = int(bag_label.item())
        model_output = model(torch.squeeze(data, 0))
        neg_values = log1mexp(torch.abs(torch.squeeze(model_output)))
        pred_0 = torch.exp(torch.sum(neg_values)).item()
        if 1 - pred_0 >= 0.5:
            pred_vals.append(1)
        else:
            pred_vals.append(0)
        true_vals.append(bag_label)

    return roc_auc_score(true_vals, pred_vals), accuracy_score(true_vals, pred_vals)

def calc_loss(model, test_loader, args):
    # calculate loss
    model.eval()
    loss = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        data, bag_label = data.float().to(device), bag_label.float().to(device)
        bag_label = int(bag_label.item())
        output = model(torch.squeeze(data, 0))
        if bag_label == 1:
            loss += (-1)*args.lw1*mil_count_loss(output, bag_label)
        else:
            loss += (-1)*args.lw2*mil_count_loss(output, bag_label)
    return float(loss/len(test_loader))

def train(args):
    print('Load Train and Test Set')
    loader_kwargs = {}

    target_number = 9
    mean_bag_length = args.mean_bag_size
    var_bag_length = args.variance_bags
    num_bags_train = args.num_train_bags
    num_bags_test = 1000

    # create train set
    train_dataset = MnistBags(target_number=target_number,
                                                mean_bag_length=mean_bag_length,
                                                var_bag_length=var_bag_length,
                                                num_bag=num_bags_train,
                                                seed=args.seed,
                                                train=True)
    
    train_loader = data_utils.DataLoader(train_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                **loader_kwargs)
    # test set
    mnist_test_bags = MnistBags(target_number=target_number,
                                                mean_bag_length=mean_bag_length,
                                                var_bag_length=var_bag_length,
                                                num_bag=num_bags_test,
                                                seed=args.seed,
                                                train=False)
    test_loader = data_utils.DataLoader(mnist_test_bags,
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)
    # test nstances
    test_instances_x, test_instances_labels = mnist_test_bags.return_data()

    # model + optimizer
    model_new = MNIST_PAPER_MODEL()
    model_new.to(device)
    optimizer = torch.optim.Adam(model_new.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    # training loop
    for epch in range(1, args.epochs + 1):
        model_new.train()
        counter = 0

        for batch_idx, (data, label) in enumerate(train_loader):
            bag_label = label[0]
            #data, bag_label = Variable(data), Variable(bag_label)
            bag_label = int(bag_label.item())
            data = data.to(device)

            # get output from model
            output = model_new(torch.squeeze(data, 0))
            if bag_label == 1:
                if counter%args.bags_per_batch == 0:
                    loss = args.lw1*(-1*mil_count_loss(output, bag_label)).clamp(min=1e-5, max=1e5)
                else:
                    loss += args.lw1*(-1*mil_count_loss(output, bag_label)).clamp(min=1e-5, max=1e5)
            else:
                if counter%args.bags_per_batch == 0:
                    loss = args.lw2*(-1*mil_count_loss(output, bag_label)).clamp(min=1e-5, max=1e5)
                else:
                    loss += args.lw2*(-1*mil_count_loss(output, bag_label)).clamp(min=1e-5, max=1e5)
            
            counter+=1
            # reset gradients
            if counter%args.bags_per_batch == 0:
                loss/=args.bags_per_batch 
                optimizer.zero_grad()
                # backward pass
                loss.backward()
                # step
                optimizer.step()

        # Calculate Classification Error/Accuracy:
        with torch.no_grad():
            # calculate metrics
            calc_auc_bag_train, calc_acc_bag_train = calculate_auc_bag_probalistic(model_new, train_loader)
            calc_loss_train = calc_loss(model_new, train_loader, args)

        print(f"Training Loss: {calc_loss_train}")
        print(f"Training Bag AUC: {calc_auc_bag_train}")
        print(f"Training Bag Acc: {calc_acc_bag_train}")
        print(f"{epch}/{args.epochs}")

    with torch.no_grad():
        # calculate final metrics
        instance_acc, instance_auc = calculate_acc_auc_instance(model_new, test_instances_x, test_instances_labels)
        calc_auc_bag_test, calc_acc_bag_test = calculate_auc_bag_probalistic(model_new, test_loader)

        # write final metrics
        with open(args.file_name, 'a') as f:
            f.write(f"Final Bag Test AUC: {calc_auc_bag_test}\n")
            f.write(f"Final Bag Test Acc: {calc_acc_bag_test}\n")
            f.write(f"Final Instance Test AUC: {instance_auc}\n")
            f.write(f"Final Instance Test Acc: {instance_acc}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regresssion')
    parser.add_argument('--mean_bag_size', type=int, default=10)
    parser.add_argument('--variance_bags', type=int, default=2)
    parser.add_argument('--num_train_bags', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--file_name', type=str, default='temp', help='file to store results')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--lw1', type=float, default=1.)
    parser.add_argument('--lw2', type=float, default=1.)
    parser.add_argument('--bags_per_batch', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_false', default=True)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(args)
    train(args)
