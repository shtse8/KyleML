import numpy as np

import torch
from torch.distributions import Categorical

# Entropy
"""
Categorical.entropy = (-array * log(array)).sum()
"""
# p = np.array([0.1, 0.2, 0.4, 0.3])
p = np.array([0.4, 0.1, 0.2, 0.2, 0.2])
p2 = p / p.sum()
print(p2)
eps = torch.finfo(torch.float).eps
print("eps = ", eps)
# (0, 1) for log
logp = np.log(np.array([max(eps, min(x, 1 - eps)) for x in p2]))
# logp = np.log(p) - np.log(np.sum(np.exp(a)))

entropy1 = np.sum(-p2*logp)
print(entropy1)
print(logp)

p_tensor = torch.Tensor(p)
dist = Categorical(probs=p_tensor)
entropy2 = dist.entropy()
print(entropy2)
print(dist.logits)
