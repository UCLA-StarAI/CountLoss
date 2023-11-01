import torch
from utils.logspace import log1mexp


def mil_count_loss(tensor_probs, pos):
    # take 1 - probability
    temp_tensor = log1mexp(torch.abs(torch.squeeze(tensor_probs)))
    # calculate the probability of 0
    sum_tensor = torch.sum(temp_tensor).clamp(max=-torch.finfo().eps)
    # calculate the probability > 0
    pos_tensor = log1mexp(torch.abs(sum_tensor))
    if pos == 1:
        return pos_tensor
    else:
        return sum_tensor



