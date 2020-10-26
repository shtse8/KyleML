import numpy as np

import torch
from torch.distributions import Categorical

# Entropy
"""
Categorical.entropy = (-array * log(array)).sum()
"""
# p = np.array([0.1, 0.2, 0.4, 0.3])
p = np.array([
    [0.4, 0.1, 0.2, 0.2, 0.2],
    [0.1, 0.2, 0.3, 0.4, 0.5]
])
m = np.array([
    [1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1]
])

def entropy_numpy(p):
    p = p / p.sum(axis=-1, keepdims=True)
    eps = torch.finfo(torch.float).eps
    logp = np.log(p.clip(eps, 1 - eps))
    entropy = np.sum(-p*logp, axis=-1)
    return entropy

def entropy_torch(p):
    entropy = Categorical(probs=p).entropy()
    return entropy

print("numpy", entropy_numpy(p))
print("torch", entropy_torch(torch.as_tensor(p, dtype=torch.float).requires_grad_(True)))


def entropy_numpy_masked(p, m):
    return np.stack([entropy_numpy(rp[rm.astype(bool)]) for rp, rm in zip(p, m)])


def entropy_torch_masked(p, m):
    return torch.stack([entropy_torch(rp[rm.to(bool)]) for rp, rm in zip(p, m)])

print("numpy_masked", entropy_numpy_masked(p, m))
print("torch_masked", entropy_torch_masked(
    torch.as_tensor(p, dtype=torch.float).requires_grad_(True), 
    torch.as_tensor(m, dtype=torch.bool)))

