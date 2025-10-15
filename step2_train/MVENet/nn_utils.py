import torch
import numpy as np

def get_l2_reg(module):
    return sum(
        torch.sum(param**2)
        for param in module.parameters()
        if param.requires_grad and param.dim() > 1
    )


def normalize(x, mean=None, std=None):
    return (x - mean) / std


def reverse_normalized(x_normalized, mean, std):
    return x_normalized * std + mean


def variance_transformation(b, numpy=True):
    if numpy:
        return np.exp(b) + 1e-6
    else:
        return torch.exp(b) + 1e-6
