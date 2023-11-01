from cgi import test
from cmath import nan
import numpy as np

import torch
from utils.logspace import *

apply_safe_log_add = SafeLogAddExp.apply


#n values
#P(x = k)
def prob_x_equals_k_final(tensor_probs, max_k, all_k, bag_size, device='cpu'):
    bags = tensor_probs.shape[0]
    rows = bag_size + 1
    cols = max_k + 1
    dp = torch.zeros(bags, rows, cols)
    dp = dp.to(device)
    dp[:,0,0] = torch.ones(bags, dtype=float).log()
    for i in range(1, rows):
        #set up tensor
        new_tensor = tensor_probs[:, i - 1]
        new_tensor = torch.unsqueeze(new_tensor, dim=1)
        new_tensor_probs = new_tensor.expand(bags, cols)
        new_tensor_minus_probs = log1mexp(-1*new_tensor_probs)
        prev_tensor = torch.add(new_tensor_minus_probs, dp[:, i - 1].clone())
        #shift previous by one to the right
        rolled_tensor = torch.roll(dp[:, i - 1].clone(), 1, 1)
        rolled_tensor[:, 0] = torch.tensor(0).log()
        prev_tensor2 = torch.add(rolled_tensor, new_tensor_probs)
        dp[:,i,:] = apply_safe_log_add(prev_tensor, prev_tensor2.clone())
    ret_tensor = dp[torch.arange(bags).long(),bag_size, all_k]
    return ret_tensor

#non-logspace implementation
def prob_x_equals_k_final_not_logspace(tensor_probs, max_k, all_k, bag_size, device='cpu'):
    bags = tensor_probs.shape[0]
    rows = bag_size + 1
    cols = max_k + 1
    dp = torch.zeros(bags, rows, cols)
    dp[:,0,0] = torch.ones(bags, dtype=float)
    dp = dp.to(device)
    for i in range(1, rows):
        #set up tensor
        new_tensor = tensor_probs[:, i - 1]
        #print(new_tensor)
        new_tensor = torch.unsqueeze(new_tensor, dim=1)
        new_tensor_probs = new_tensor.expand(bags, cols)
        new_tensor_minus_probs = 1 - new_tensor_probs
        prev_tensor = torch.multiply(new_tensor_minus_probs, dp[:, i - 1].clone())
        #shift previous by one to the right
        rolled_tensor = torch.roll(dp[:, i - 1].clone(), 1, 1)
        rolled_tensor[:, 0] = torch.tensor(0)
        prev_tensor2 = torch.multiply(rolled_tensor, new_tensor_probs)
        dp[:,i] = torch.add(prev_tensor, prev_tensor2.clone())
    ret_tensor = dp[torch.arange(bags).long(),bag_size, all_k.long()]
    return ret_tensor



#returns the entire tensor for a maximum k value
def ret_entire_tensor_k(tensor_probs, max_k):

    bags = tensor_probs.shape[0]
    rows = tensor_probs.shape[1] + 1
    cols = max_k + 1
    dp = torch.zeros(bags, rows, cols).log()
    dp[:,0,0] = torch.ones(bags, dtype=float).log()
    for i in range(1, rows):
        #set up tensor
        new_tensor = tensor_probs[:, i - 1]
        new_tensor = torch.unsqueeze(new_tensor, dim=1)
        new_tensor_probs = new_tensor.expand(bags, cols)

        new_tensor_minus_probs = log1mexp(torch.abs(new_tensor_probs))
        prev_tensor = torch.add(new_tensor_minus_probs, dp[:, i - 1].clone())
        #shift previous by one to the right
        rolled_tensor = torch.roll(dp[:, i - 1].clone(), 1, 1)
        rolled_tensor[:, 0] = torch.tensor(0).log()
        prev_tensor2 = torch.add(rolled_tensor, new_tensor_probs)
        dp[:,i] = apply_safe_log_add(prev_tensor, prev_tensor2.clone())
    return dp
