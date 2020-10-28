import torch
import torch.nn as nn
import torch.nn.functional as F
import math

x = torch.ones(2, 4)
linear = nn.Linear(4, 4)
# def init(m):
#     nn.init.constant_(m.weight, 0)
#     nn.init.constant_(m.bias, 0)
# linear.apply(init)
p = linear(x)

m = torch.ones_like(p, dtype=torch.bool)
print("mask", m)
mp = F.softmax(p.masked_fill(~m, -math.inf), dim=-1)
loss = -torch.distributions.Categorical(probs=mp).entropy().mean()
print(loss)
loss.backward(retain_graph=True)
print(linear.weight.grad)
linear.weight.grad.data.zero_()

print("=================== masked -inf")
m = torch.ones_like(p, dtype=torch.bool)
m[0][0] = False
m[0][1] = False
m[0][2] = False
m[0][3] = True
print("mask", m)
mp = F.softmax(p.masked_fill(~m, -math.inf), dim=-1)
print(mp)
loss = -torch.distributions.Categorical(probs=mp).entropy().mean()
print(loss)
loss.backward(retain_graph=True)
print(linear.weight.grad)
linear.weight.grad.data.zero_()
print("=================== masked 0")
m = torch.ones_like(p, dtype=torch.bool)
m[0][0] = False
m[0][1] = False
m[0][2] = False
m[0][3] = True
print("mask", m)
mp = F.softmax(p, dim=-1).masked_fill(~m, 0)
mp = mp / mp.sum(dim=-1, keepdims=True)
print(mp)
loss = -torch.distributions.Categorical(probs=mp).entropy().mean()
print(loss)
loss.backward(retain_graph=True)
print(linear.weight.grad)
linear.weight.grad.data.zero_()

print("=================== where")
mp = F.softmax(p, dim=-1)
print(mp)
loss = -torch.stack([torch.distributions.Categorical(probs=rp[rm]).entropy() for rp, rm in zip(mp, m)]).mean()
print(loss)
loss.backward(retain_graph=True)
print(linear.weight.grad)
linear.weight.grad.data.zero_()