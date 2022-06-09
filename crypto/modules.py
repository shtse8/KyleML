import numpy as np
import torch
from torch import nn


class SwitchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997, using_moving_average=True):
        super(SwitchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.mean_weight = nn.Parameter(torch.ones(2))
        self.var_weight = nn.Parameter(torch.ones(2))
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.zeros(1, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        mean_ln = x.mean(1, keepdim=True)
        var_ln = x.var(1, keepdim=True)

        if self.training:
            mean_bn = x.mean(0, keepdim=True)
            var_bn = x.var(0, keepdim=True)
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean.add_(mean_bn.data)
                self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        mean = mean_weight[0] * mean_ln + mean_weight[1] * mean_bn
        var = var_weight[0] * var_ln + var_weight[1] * var_bn

        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight + self.bias


class LinearEx(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearEx, self).__init__()
        self.layer = nn.Linear(in_size, out_size)

    def forward(self, x):
        size_length = len(x.size())
        # (N, C)
        if size_length == 2:
            return self.layer(x)
        # (N, L, C)
        elif size_length == 3:
            batch_size, seq_len, _ = x.size()
            x = x.view(batch_size * seq_len, -1)
            x = self.layer(x)
            x = x.view(batch_size, seq_len, -1)
            return x
        else:
            raise ValueError(f"Invalid input size, expected (N, C) or (N, L, C), but got {x.size()}")


class BatchNorm1dEx(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm1dEx, self).__init__()
        self.layer = nn.BatchNorm1d(num_features)

    def forward(self, x):
        size_length = len(x.size())
        # (N, C)
        if size_length == 2:
            return self.layer(x)
        # (N, L, C)
        elif size_length == 3:
            # (N, L, C) => (N, C, L)
            x = x.permute(0, 2, 1)

            # nn.BatchNorm1d only support (N, C, L)
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
            x = self.layer(x)

            # (N, C, L) => (N, L, C)
            x = x.permute(0, 2, 1)
            return x
        else:
            raise ValueError(f"Invalid input size, expected (N, C) or (N, L, C), but got {x.size()}")
