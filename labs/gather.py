import numpy as np

import torch
from torch.distributions import Categorical

# Entropy
"""
Categorical.entropy = (-array * log(array)).sum()
"""
# p = np.array([0.1, 0.2, 0.4, 0.3])
# p = np.array([
#     [[0, 1], [2, 3], [4, 5], [6, 7]],
    # [[1, 2], [3, 4], [5, 6], [7, 8]]])
p = np.array([
    [[[0, 1], [0, 1]], [[2, 3], [2, 3]]],
    [[[1, 2], [1, 2]], [[3, 4], [3, 4]]]])

# p = np.array([
#     [0, 1, 2, 3],
#     [2, 3, 4, 5]])
i = np.array([0, 1])

p = torch.tensor(p).flatten(2)
i = torch.tensor(i, dtype=torch.long)
print(i)
print(p.size(), i.size())
print(p[torch.arange(p.size(0)), i])
