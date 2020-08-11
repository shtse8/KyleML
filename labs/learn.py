
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

modelA = nn.Linear(10, 10)
modelB = nn.Linear(10, 10)
modelC = nn.Linear(10, 10)

x = torch.randn(1, 10)
print(x)

a = modelA(x)
b = modelB(a.detach())
b.mean().backward()
print("modelA.weight.grad", modelA.weight.grad)
print("modelB.weight.grad", modelB.weight.grad)
print("modelC.weight.grad", modelC.weight.grad)

c = modelC(a)
c.mean().backward()
print(modelA.weight.grad)
print(modelB.weight.grad)
print(modelC.weight.grad)