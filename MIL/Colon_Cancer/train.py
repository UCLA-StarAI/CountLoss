from __future__ import print_function

import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import argparse
import random

import numpy as np
import torch
import torch.utils.data as data_utils
from colon_utils.dataset import *
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from utils.colon_models import *
from utils.data_processor import *
from utils.dataloader import *
from utils.log_space_algo import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_acc_bag_probalistic(model, test_loader):
    # calculate bag level metrics
    model.eval()
    pred_vals = []
    true_vals = []
    for batch_idx, (data, label) in enumerate(test_loader):
        # get data
        bag_label = label[0]
        data, bag_label = data.float().to(device), bag_label.float().to(device)

        # pass through model
        model_output = model(torch.squeeze(data))

        # take all 1 - probs
        neg_values = log1mexp(torch.abs(torch.squeeze(model_output)))
        
        # get probability that it is 0
        pred_0 = torch.exp(torch.sum(neg_values)).item()
        if 1 - pred_0 >= 0.5:
            pred_vals.append(1)
        else:
            pred_vals.append(0)
        true_vals.append(bag_label.float().item())
    
    return accuracy_score(true_vals, pred_vals), roc_auc_score(true_vals, pred_vals), f1_score(true_vals, pred_vals), precision_score(true_vals, pred_vals), recall_score(true_vals, pred_vals)

def calc_loss(model, test_loader):
    # calculate loss
    model.eval()
    loss = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        # get data
        bag_label = label[0]
        data, bag_label = data.float().to(device), bag_label.float().to(device)
        bag_label = int(bag_label.item())

        # pass through model
        output = model(torch.squeeze(data))

        # calculate loss
        loss = loss + (-1)*mil_count_loss(torch.squeeze(output), bag_label)
    return float(loss/test_loader.__len__())

def train(args):
    print('Load Train and Test Set')
    ret_datasets = load_dataset(dataset_path='Patches', n_folds=10, rand_state=args.seed)
    all_fold_vals_acc = []
    all_fold_vals_auc = []
    all_fold_vals_f1 = []
    all_fold_vals_pre = []
    all_fold_vals_rec = []

    for ifold in range(10):
        f = open(args.file_name, "a")
        f.write(f"Fold {ifold} Start Training: \n")
        f.close()
        train_bags = generate_batch(ret_datasets[ifold]['train'])
        test_bags = generate_batch(ret_datasets[ifold]['test'])

        # create datasets
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
        train_dataset = Colon_Dataset(train_bags)
        train_loader = data_utils.DataLoader(train_dataset,
                                                batch_size=1,
                                                shuffle=True,
                                                **loader_kwargs)
        mnist_test_bags = Colon_Dataset(test_bags)
        test_loader = data_utils.DataLoader(mnist_test_bags,
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)
        # create model
        model_new = Colon_Models()
        model_new.to(device)

        # create optimizer
        optimizer = torch.optim.Adam(model_new.parameters(), lr=args.lr, weight_decay=args.L2, betas=(0.9, 0.999))
        
        # iterate through
        for epch in range(1, args.epochs + 1):
            model_new.train()
            avg_loss = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                # get data
                bag_label = label[0]
                data, bag_label = data.float().to(device), bag_label.float().to(device)
                bag_label = int(bag_label.item())

                # reset gradients
                optimizer.zero_grad()

                # get output from model
                output = model_new(torch.squeeze(data))
                
                # compute loss
                if bag_label == 1:
                    loss = args.lw1*mil_count_loss(output, bag_label).clamp(max=-torch.finfo().eps)
                else:
                    loss = args.lw2*mil_count_loss(output, bag_label).clamp(max=-torch.finfo().eps)
                
                loss = loss*(-1)
                avg_loss += loss
                # backward pass
                loss.backward()
                # step
                optimizer.step()
            

            avg_loss/=(batch_idx + 1)
            print("Epoch: ", epch)
            print("Loss: ", avg_loss.item())

            with torch.no_grad():

                # calculate eval metrics
                model_new.eval()
                calc_acc_bag_train, _, _, _, _ = calculate_acc_bag_probalistic(model_new, train_loader)
                calc_loss_train = calc_loss(model_new, train_loader)

                f = open(args.file_name, "a")
                f.write(f"Epoch: {epch}\n")
                f.write(f"Training Loss: {avg_loss}\n")
                f.write(f"Training Acc: {calc_acc_bag_train}\n")
                f.close()
        # add last value
        calc_acc_bag_test, calc_auc_bag_test, calc_f1_bag_test, calc_precision_bag_test, calc_recall_bag_test = calculate_acc_bag_probalistic(model_new, test_loader)
        all_fold_vals_acc.append(calc_acc_bag_test)
        all_fold_vals_auc.append(calc_auc_bag_test)
        all_fold_vals_f1.append(calc_f1_bag_test)
        all_fold_vals_pre.append(calc_precision_bag_test)
        all_fold_vals_rec.append(calc_recall_bag_test)
        f = open(args.file_name, "a")
        f.write(f'{calc_acc_bag_test}\n')
        f.write(f'{calc_auc_bag_test}\n')
        f.write(f'{calc_f1_bag_test}\n')
        f.write(f'{calc_precision_bag_test}\n')
        f.write(f'{calc_recall_bag_test}\n')

        f.close()
    f = open(args.file_name, "a")
    f.write(f"Final mean acc: {np.mean(np.array(all_fold_vals_acc))}\n")
    f.write(f"Final mean auc: {np.mean(np.array(all_fold_vals_auc))}\n")
    f.write(f"Final mean f1: {np.mean(np.array(all_fold_vals_f1))}\n")
    f.write(f"Final mean precision: {np.mean(np.array(all_fold_vals_pre))}\n")
    f.write(f"Final mean recall: {np.mean(np.array(all_fold_vals_rec))}\n")

    f.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colon Cancer dataset Regresssion')
    parser.add_argument('--L2', type=float, default=5e-4)
    parser.add_argument('--lw1', type=float, default=1.)
    parser.add_argument('--lw2', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed',type=int, default=1)
    parser.add_argument('--file_name',type=str,default='temp')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train(args)