import numpy as np

import torch
from torch.distributions import Categorical
import time

# Entropy
"""
Categorical.entropy = (-array * log(array)).sum()
"""
# p = np.array([0.1, 0.2, 0.4, 0.3])
p = np.random.randn(2, 4)
m = p > 0
p = np.abs(p)

def printit(i, *s):
    if i == 1:
        print(*s)

def timeit(func):
    stime = time.perf_counter()
    result = func()
    duration = time.perf_counter() - stime
    return str(result) + f" (time: {duration})"

def entropy_numpy(p, m = None):
    if m is not None:
        p = p * m
    p = p / p.sum(axis=-1, keepdims=True)
    eps = torch.finfo(torch.float).eps
    logp = np.log(p.clip(eps, 1 - eps))
    entropy_values = -p*logp
    if m is not None:
        entropy_values = entropy_values * m
    entropy = np.sum(entropy_values, axis=-1)
    return entropy

def entropy_torch(p, m = None):
    if m is not None:
        p = p.masked_fill(~m, 0)
    p = p / p.sum(axis=-1, keepdims=True)
    eps = torch.finfo(torch.float).eps
    logp = p.clamp(eps, 1 - eps).log()
    entropy_values = -p*logp
    if m is not None:
        entropy_values = entropy_values.masked_fill(~m, 0)
    entropy = torch.sum(entropy_values, axis=-1)
    return entropy

def entropy_numpy_masked(p, m):
    return entropy_numpy(p, m)
    # return np.stack([entropy_numpy(rp[rm.astype(bool)]) for rp, rm in zip(p, m)])
    # return np.stack([entropy_numpy(rp[rm.astype(bool)]) for rp, rm in zip(p, m)])


def entropy_torch_masked(p, m):
    return entropy_torch(p, m)
    # return torch.stack([entropy_torch(rp[rm.type(torch.bool)]) for rp, rm in zip(p, m)])

for i in range(2):
    printit(i, "numpy", timeit(lambda: entropy_numpy(p).mean()))
    printit(i, "torch", timeit(lambda: entropy_torch(torch.as_tensor(p, dtype=torch.float, device=torch.device('cuda')).requires_grad_(True)).mean()))
    printit(i, "numpy_masked", timeit(lambda: entropy_numpy_masked(p, m).mean()))
    printit(i, "torch_masked", timeit(lambda: entropy_torch_masked(
        torch.as_tensor(p, dtype=torch.float, device=torch.device('cuda')).requires_grad_(True), 
        torch.as_tensor(m, dtype=torch.bool, device=torch.device('cuda'))).mean()))

