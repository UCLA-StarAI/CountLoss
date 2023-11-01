import os

# set number of threads to use
os.environ["OMP_NUM_THREADS"] = "15" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "15" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "15" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "15" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "15" # export NUMEXPR_NUM_THREADS=1

import argparse
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
# utility packages
from utils.dataset import *
from utils.log_space_algo import *
from utils.logspace import *
from utils.models import *
from utils.preprocessing import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(1)
torch.manual_seed(1)

def get_dataset(args):
    # use the code from LMMCM
    train_X, train_y, test_X, test_y, bag_id, prop_dict, size_dict, cont_col = parse_llp_data(args)
    train_X, test_X = feature_engineering_cont(train_X, test_X, cont_col)

    # convert pd dataframes to lists and tensors
    train_y = torch.tensor(train_y.values.tolist())
    train_X = torch.tensor(train_X.values.tolist())
    test_X = torch.tensor(test_X.values.tolist())
    test_y = [0 if test_label == -1 else 1 for test_label in test_y.values.tolist()]
    test_y = torch.tensor(test_y)

    # create bags
    training_tensors, prop_labels, count_labels = create_bags(train_X, train_y, bag_id, prop_dict)
    
    # create training and test sset
    data_set = Dataset_Bags(training_tensors, prop_labels, count_labels)
    
    val_X = None
    val_y = None
    # create val set
    if args.val_set:
        num_val = int(len(training_tensors)*.125)
        training_set, validation_set = torch.utils.data.random_split(data_set, [len(training_tensors) - num_val, num_val])
        val_X, _, val_y = validation_set._return_all()
    else:
        training_set = data_set

    # create train data loaders
    params = {'batch_size': args.batch_size, 'shuffle': True}
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    return training_generator, val_X, val_y, test_X, test_y

def evaluate(model, X, y):
    model.eval()
    model_preds = torch.squeeze(model(X))
    pred_X = torch.exp(model_preds).round()
    calc_loss = -1*(prob_x_equals_k_final(model_preds, max(y), y.numpy(), args.bag_size))
    return roc_auc_score(y, pred_X), calc_loss


def prop_loss(ground_truth_prop, output):
    loss = torch.nn.BCELoss()
    return loss(torch.mean(output, 1).float(), ground_truth_prop.float())

def train(args):
    print("Train Loop Beginning...")

    # get train generator
    training_generator, val_X, val_y, test_X, test_y = get_dataset(args)

    # get model
    if args.dataset_name == 'magic':
        model = Magic_Model(10)
    elif args.dataset_name == 'adult':
        model = Adult_Model(108)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)

    # warm up epochs
    for warm_epoch in range(1, args.min_epochs + 1):
        model.train()
        avg_loss = 0.
        for train_batch, prop_labels, count_labels in training_generator:
            optimizer.zero_grad()

            # get output from model
            outputs = model(train_batch)

            # compute exactly-k loss
            if args.loss == 'count':
                loss = -1*(prob_x_equals_k_final(outputs, max(count_labels), count_labels.numpy(), args.bag_size))
            else:
                # note that we can use a model with sigmoid output not logsigmoid too
                # but we do this for simplicity
                loss = prop_loss(prop_labels, torch.exp(model))
            loss = loss.mean()
            avg_loss += loss.item()

            # L1 regularization
            if args.use_L1:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l1_lambda = args.L1
                loss += l1_lambda*l1_norm

            loss.backward()
            optimizer.step()
        print(f"Warm Start Epoch {warm_epoch}/{args.min_epochs}")
        print(f"Loss: {avg_loss}")
    # early stopping counter
    counter_early = 0
    # early stopping patience
    patience = 500

    # previous
    val_loss = np.inf
    
    # final metrics
    final_test_auc = None

    # main training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        avg_loss = 0.
        for train_batch, prop_labels, count_labels in training_generator:
            optimizer.zero_grad()

            # get output from model
            outputs = torch.squeeze(model(train_batch))

            # compute exactly-k loss
            if args.loss == 'count':
                loss = -1*(prob_x_equals_k_final(outputs, max(count_labels), count_labels.numpy(), args.bag_size))
            else:
                # note that we can use a model with sigmoid output not logsigmoid too
                # but we do this to make code smaller
                loss = prop_loss(prop_labels, torch.exp(model))
            loss = loss.mean()
            avg_loss += loss.item()

            # L1 regularization
            if args.use_L1:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l1_lambda = args.L1
                loss += l1_lambda*l1_norm

            loss.backward()
            optimizer.step()
        print(f"Training Epoch {epoch}/{args.epochs}")
        print(f"Loss: {avg_loss}")
        # compute metrics on training data
        if args.val_set:
            _, curr_val_loss = evaluate(model, val_X, val_y)
            if curr_val_loss < val_loss:
                final_test_auc, _ = evaluate(model, test_X, test_y)
                counter_early = 0
            else:
                counter_early+=1
            if counter_early == patience:
                break
    #calculate metric at end
    final_test, _ = evaluate(model, test_X, test_y)

    # write metrics to file
    with open(args.write_path, 'a') as f:
        f.write(f"Final Results:\n")
        f.write(f"Final test AUC based on val loss: {final_test}")
        f.write(f"Final AUC test: {final_test_auc}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLP :)')
    parser.add_argument('--bag_size', type=int, default=8)
    parser.add_argument('--log_dir', type=str, default="test_dir")
    parser.add_argument('--layer_sizes', nargs="+", default=[2048])
    parser.add_argument('--L1', type=float, default=1e-3)
    parser.add_argument('--use_L1', action='store_true')
    parser.add_argument('--L2', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--min_epochs', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="experiments_adult_0_one")
    parser.add_argument('--dataset_name', type=str, default="adult")
    parser.add_argument('--test_num', type=int, default=0)
    parser.add_argument('--write_path', type=str, default='temp')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_set', action='store_true')
    parser.add_argument('--loss', type=str, default='count')
    args = parser.parse_args()

    print(args)
    with open(args.write_path, 'a') as f:
        f.write(f"{args}\n")
    train(args)
