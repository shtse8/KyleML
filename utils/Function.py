import sys
import torch


def normalize(value):
    # print(sys.float_info.min, torch.finfo(torch.float).eps)
    eps = sys.float_info.min
    if isinstance(value, torch.Tensor):
        eps = torch.finfo(torch.float).eps
    value = (value - value.mean()) / (value.std() + eps)
    return value
