import sys
import torch

"""
    數組為正態分佈
    計算出最後每個數值的標準差
"""
def normalize(value):
    # print(sys.float_info.min, torch.finfo(torch.float).eps)
    eps = sys.float_info.min
    if isinstance(value, torch.Tensor):
        eps = torch.finfo(torch.float).eps
    value = (value - value.mean()) / (value.std() + eps)
    return value
