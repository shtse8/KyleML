import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as schedular
from enum import Enum

from .Message import NetworkInfo


class Activator(Enum):
    ReLU = 0
    ELU = 1


class Norm(Enum):
    BatchNorm = 0
    GroupNorm = 1


class Norm2dLayer(nn.Module):
    def __init__(self, num_features, type):
        super().__init__()
        self.layer = lambda x: x
        if type == Norm.BatchNorm:
            self.layer = nn.BatchNorm2d(num_features)
        elif type == Norm.GroupNorm:
            raise Exception("Group norm is not supported.")

    def forward(self, x):
        return self.layer(x)


class Norm1dLayer(nn.Module):
    def __init__(self, num_features, norm_type):
        super().__init__()
        self.layer = lambda x: x
        if norm_type == Norm.BatchNorm:
            self.layer = nn.BatchNorm1d(num_features)
        elif norm_type == Norm.GroupNorm:
            raise Exception("Group norm is not supported.")

    def forward(self, x):
        return self.layer(x)


class ActivatorLayer(nn.Module):
    def __init__(self, norm_type):
        super().__init__()
        self.layer = lambda x: x
        if norm_type == Activator.ELU:
            self.layer = nn.ELU()
        elif norm_type == Activator.ReLU:
            self.layer = nn.ReLU()

    def forward(self, x):
        return self.layer(x)


class ConvLayers(nn.Module):
    def __init__(self, input_shape, hidden_size, activator: Activator = Activator.ELU, norm: Norm = Norm.BatchNorm):
        super().__init__()
        if min(input_shape[1], input_shape[2]) < 20:
            
            # small CNN
            self.layers = nn.Sequential(
                nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
                ActivatorLayer(activator),
                Norm2dLayer(16, norm),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                ActivatorLayer(activator),
                Norm2dLayer(32, norm),
                nn.Flatten(),
                nn.Linear(32 * input_shape[1] * input_shape[2], hidden_size),
                ActivatorLayer(activator),
                Norm1dLayer(hidden_size, norm))
        else:
            self.layers = nn.Sequential(
                # [C, H, W] -> [32, H, W]
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                ActivatorLayer(activator),
                Norm2dLayer(32, norm),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                ActivatorLayer(activator),
                Norm2dLayer(64, norm),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                ActivatorLayer(activator),
                Norm2dLayer(64, norm),
                # [64, H, W] -> [64 * H * W]
                nn.Flatten(),
                nn.Linear(64 * input_shape[1] * input_shape[2], hidden_size),
                ActivatorLayer(activator),
                Norm1dLayer(hidden_size, norm))
        self.num_output = hidden_size

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)  # [B, H, W, C] => [B, C, H, W]
        return self.layers(x)


class FCLayers(nn.Module):
    def __init__(self, n_inputs, hidden_size, num_layers=1, activator: Activator = Activator.ELU, norm: Norm = Norm.BatchNorm):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_nodes = n_inputs if i == 0 else hidden_size
            self.layers.append(nn.Linear(in_nodes, hidden_size))
            self.layers.append(ActivatorLayer(activator))
            self.layers.append(Norm1dLayer(hidden_size, norm))
        self.num_output = hidden_size

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

class BodyLayers(nn.Module):
    def __init__(self, inputShape, hidden_nodes, activator: Activator = Activator.ELU, norm: Norm = Norm.BatchNorm):
        super().__init__()
        if type(inputShape) is tuple and len(inputShape) == 3:
            self.layers = ConvLayers(inputShape, hidden_nodes, activator=activator, norm=norm)
        else:
            if type(inputShape) is tuple and len(inputShape) == 1:
                inputShape = inputShape[0]
            self.layers = FCLayers(inputShape, hidden_nodes, num_layers=3, activator=activator, norm=norm)
        self.num_output = hidden_nodes
            
    def forward(self, x):
        return self.layers(x)


class Network(nn.Module):
    def __init__(self, input_shape, n_outputs, name="network"):
        super().__init__()
        self.name = name
        self.optimizer = None
        self.version: int = 1
        self.info: NetworkInfo = None

    def buildOptimizer(self):
        raise NotImplementedError

    @staticmethod
    def initWeight(m):
        if type(m) == nn.GRU:
            # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L91
            for key, value in m.named_parameters():
                if 'weight' in key:
                    # print("initializing:", type(m).__name__, key)
                    nn.init.orthogonal_(value)
                elif 'bias' in key:
                    nn.init.constant_(value, 0)
        elif type(m) in [nn.Linear, nn.Conv2d]:
            # print("initializing:", type(m).__name__)
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def initWeights(self):
        self.apply(Network.initWeight)

    def _updateStateDict(self):
        if self.info is None or self.info.version != self.version:
            # print("Update Cache", self.version)
            stateDict = self.state_dict()
            for key, value in stateDict.items():
                stateDict[key] = value.cpu()  # .detach().numpy()
            self.info = NetworkInfo(stateDict, self.version)

    def getInfo(self) -> NetworkInfo:
        self._updateStateDict()
        return self.info

    def loadInfo(self, info: NetworkInfo):
        stateDict = info.stateDict
        # for key, value in stateDict.items():
        #     stateDict[key] = torch.from_numpy(value)
        self.load_state_dict(stateDict)
        self.version = info.version

    def isNewer(self, info: NetworkInfo):
        return info.version > self.version

class GRULayers(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

    @property
    def num_directions(self):
        return 2 if self.gru.bidirectional else 1

    @property
    def num_output(self):
        return self.gru.hidden_size * self.num_directions

    def getInitHiddenState(self, device):
        return torch.zeros((self.gru.num_layers * self.num_directions, self.gru.hidden_size), device=device)

    def forward(self, x, h):
        # x: (B, N) -> (1, B, N)
        # h: (B, L, H) -> (L, B, H)
        x, h = self.gru(x.unsqueeze(0), h.transpose(0, 1).contiguous())
        # x: (1, B, N) -> (B, N)
        # h: (L, B, H) -> (B, L, H)
        return x.squeeze(0), h.transpose(0, 1)
