import numpy as np

import torch
from torch.distributions import Categorical

# Entropy
"""
Categorical.log_prob = a.gather(b).log()
"""
p = np.array([0.1, 0.2, 0.4, 0.3])
a = np.array([0])
logp = np.log(p[a])
print(logp)

entropy2 = Categorical(probs = torch.Tensor(p)).log_prob(torch.Tensor(a))
print(entropy2)