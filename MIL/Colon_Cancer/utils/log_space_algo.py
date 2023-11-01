import torch


def log1mexp(x):
    assert(torch.all(x >= 0))
    return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

def mil_count_loss(tensor_probs, pos):
    temp_tensor = log1mexp(torch.abs(torch.squeeze(tensor_probs)))
    sum_tensor = torch.sum(temp_tensor).clamp(max=-torch.finfo().eps)
    pos_tensor = log1mexp(torch.abs(sum_tensor))
    if pos == 1:
        return pos_tensor
    else:
        return sum_tensor

