import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

s = torch.FloatTensor(np.array([0.1, 0.2, 0.4, 0.3]))


print(s.unsqueeze(1).squeeze(1))

t = torch.FloatTensor(np.array([1, 2, 3, 4]))
print(s, t)
loss = nn.MSELoss()(s.unsqueeze(1), t.unsqueeze(1))
print(loss)
loss = nn.MSELoss()(s, t.unsqueeze(1))
print(loss)
loss = nn.MSELoss()(s.unsqueeze(1), t)
print(loss)
loss = nn.MSELoss()(s, t)
print(loss)

loss = (t.unsqueeze(1) - s.unsqueeze(1)).pow(2).mean()
print(loss)
loss = (t.unsqueeze(1) - s).pow(2).mean()
print(loss)
loss = (t - s.unsqueeze(1)).pow(2).mean()
print(loss)
loss = (t - s).pow(2).mean()
print(loss)