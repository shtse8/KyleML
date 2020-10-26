import torch


class Categorical:
    def __init__(self, probs, masks=None):
        self.probs = probs
        self.masks = masks
        if masks is not None:
            self.probs = self.probs.masked_fill(~self.masks, 0)
        self.probs = self.probs / self.probs.sum(axis=-1, keepdims=True)
    
    @property
    def logits(self):
        eps = torch.finfo(torch.float).eps
        return self.probs.clamp(eps, 1 - eps).log()

    def entropy(self):
        values = self.probs * self.logits
        if self.masks is not None:
            values = values.masked_fill(~self.masks, 0)
        return -values.sum(axis=-1)
