import numpy as np
import torch
from count_loss_funcs.loss_func import *


def p_loss_func(output, num_pos_instances, loss_weight):
    if num_pos_instances > 0:
        loss2 = (-1*torch.sum(output))/num_pos_instances
        if not torch.isnan(loss2) and not torch.isinf(loss2):
            return loss2

def u_loss_func(output, global_prop, num_u_instances, device):
    prop_val = num_u_instances*global_prop
    prop_val = round(prop_val)
    return -1*prob_x_equals_k_final(output, prop_val, [prop_val], num_u_instances, device)

def u_loss_dist(output, prop, num_u_instances, device):
    return dist_loss(output, num_u_instances, prop, device)
