import pickle

import numpy as np
import torch
from sklearn import preprocessing


def parse_llp_data(args):
    num = 0
    data_path = f'Learning_Label_Prop/LMMCM/{args.dataset}/{args.dataset_name}{num}_{args.bag_size}_{args.test_num}'
    with open(data_path, 'rb') as data_file:
        data_dict = pickle.load(data_file)
    #pd tables

    #8192 x 108
    train_X = data_dict['train_X']
    #8192 x 1
    train_y = data_dict['train_y']
    #16280 x 108
    test_X = data_dict['test_X']
    #16280 x 1
    test_y = data_dict['test_y']

    #np array of bag ids
    bag_id = data_dict['bag_id']

    #dictionary of probability of each index
    prop_dict = data_dict['prop_dict']

    #maps index to size
    size_dict = data_dict['size_dict']
    
    k_fold = data_dict['k_fold']
    cont_col = data_dict['cont_col']
    bag_to_fold = data_dict['bag_to_fold']

    extra_tests = data_dict['extra_tests']

    return train_X, train_y, test_X, test_y, bag_id, prop_dict, size_dict, cont_col

def feature_engineering_cont(train_X, test_X, cont_col):
	# Arguments:
	#     train_X: training feature matrix
	# 	  test_X: testing feature matrix
	#     cat_col: the list of columns with continuous values
	#
	# Functionality:
	#     standardize all continuous values;
	#
	# Returns:
	#     train_X: the engineered training feature matrix
	#     test_X: the engineered testing feature matrix

	# standardize continuous features - also include dates
	train_X_after = train_X.copy()
	test_X_after = test_X.copy()
	if len(cont_col) != 0:
		scaler = preprocessing.StandardScaler()
		train_X_after.loc[:, cont_col] = scaler.fit_transform(train_X_after[cont_col])
		test_X_after.loc[:, cont_col] = scaler.transform(test_X_after[cont_col])
	return train_X_after, test_X_after

def create_bags(train_X, train_y, bag_id, prop_dict):
    print("Create bags")

    # assigns instance indexes to bags
    bag_id_to_instance_index = {}
    for i in range(len(bag_id)):
        id = bag_id[i]
        if id in bag_id_to_instance_index.keys():
            bag_id_to_instance_index[id].append(i)
        else:
            bag_id_to_instance_index[id] = [i]
    
    # Make bags into torch tensors
    training_tensors = []
    prop_labels = []
    count_labels = []

    for bag_id in bag_id_to_instance_index:
        # label as a proportion
        prop_labels.append(prop_dict[bag_id])

        accum_tensor = None
        tot_val = 0

        for tensor in bag_id_to_instance_index[bag_id]:
            if train_y[tensor] == 1:
                tot_val += 1
            if accum_tensor == None:
                accum_tensor = train_X[tensor]
                accum_tensor = accum_tensor.unsqueeze(dim=0)
            else:
                accum_tensor = torch.cat([accum_tensor, train_X[tensor].unsqueeze(dim=0)], dim=0)
        #append training tensor and count to list
        training_tensors.append(accum_tensor)
        count_labels.append(tot_val)
    return training_tensors, prop_labels, count_labels