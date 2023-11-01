from cgi import test
from cmath import nan

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom


class SafeLogAddExp(torch.autograd.Function):
    """Implements a torch function that is exactly like logaddexp,
    but is willing to zero out nans on the backward pass."""

    @staticmethod
    def forward(ctx, input, other):
        with torch.enable_grad():
            output = torch.logaddexp(input, other) # internal copy of output
        ctx.save_for_backward(input, other, output)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, other, output = ctx.saved_tensors
        grad_input, grad_other = torch.autograd.grad(output, (input, other), grad_output, only_inputs=True)
        mask = torch.isinf(input).logical_and(input == other)
        grad_input[mask] = 0
        grad_other[mask] = 0
        return grad_input, grad_other

logaddexp = SafeLogAddExp.apply

def log1mexp(x):
    assert(torch.all(x >= 0))
    return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

apply_func = SafeLogAddExp.apply

def prob_x_equals_k_final(tensor_probs, max_k, all_k, bag_size, device='cpu'):
    """
    Exactly-k loss function
    """
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
        dp[:,i,:] = apply_func(prev_tensor, prev_tensor2.clone())
    ret_tensor = dp[torch.arange(bags).long(),bag_size, all_k]
    return ret_tensor

def ret_entire_tensor_k(tensor_probs, max_k, device='cpu'):
    """
    returns entire count distribution
    """
    bags = tensor_probs.shape[0]
    rows = tensor_probs.shape[1] + 1
    cols = max_k + 1
    dp = torch.zeros(bags, rows, cols).log()
    dp[:,0,0] = torch.ones(bags, dtype=float).log()
    dp = dp.to(device)
    for i in range(1, rows):
        #set up tensor
        new_tensor = tensor_probs[:, i - 1]
        #print(new_tensor)
        new_tensor = torch.unsqueeze(new_tensor, dim=1)
        new_tensor_probs = new_tensor.expand(bags, cols)

        #new_tensor_minus_probs = log1mexp(-1*new_tensor_probs)
        new_tensor_minus_probs = log1mexp(torch.abs(new_tensor_probs))
        prev_tensor = torch.add(new_tensor_minus_probs, dp[:, i - 1].clone())
        #shift previous by one to the right
        rolled_tensor = torch.roll(dp[:, i - 1].clone(), 1, 1)
        rolled_tensor[:, 0] = torch.tensor(0).log()
        #print(prev_tensor)
        prev_tensor2 = torch.add(rolled_tensor, new_tensor_probs)
        #print(prev_tensor2)
        dp[:,i] = apply_func(prev_tensor, prev_tensor2.clone())
        #print(dp)
    return dp

def dist_loss(tensor_probs, bag_size, prop, device):
    get_all_vals = ret_entire_tensor_k(tensor_probs, bag_size, device)
    get_final_vals = get_all_vals[0, bag_size, :]
    binomial_pmf = torch.tensor(binom.pmf(np.arange(0, bag_size + 1), bag_size, prop)).to(device).log()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    score = kl_loss(get_final_vals, binomial_pmf)
    return score

