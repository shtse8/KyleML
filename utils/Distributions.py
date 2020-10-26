import torch


class Categorical:
    def __init__(self, probs, masks=None):
        self.probs = probs
        self.masks = masks
        
    def entropy(self):
        if self.masks is not None:
            return torch.stack([torch.distributions.Categorical(probs=p[m.to(bool)]).entropy() for p, m in zip(self.probs, self.masks)])
        else:
            return torch.distributions.Categorical(probs=self.probs).entropy()