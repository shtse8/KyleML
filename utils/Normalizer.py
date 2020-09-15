import numpy as np
import math


class Normalizer:
    def __init__(self, epsilon=1e-8, shape=()):
        pass

    def update(self, x):
        pass

    def normalize(self, x, update=False):
        pass


class StdNormalizer(Normalizer):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-8, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 0
        self.epsilon = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count

        self.count = total_count

    def normalize(self, x, update=False):
        if update:
            self.update(x)
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        
    def __str__(self):
        return f"StdNormalizer(mean={self.mean:.4f};var={self.var:.4f};count={self.count})"


class RangeNormalizer(Normalizer):
    def __init__(self, epsilon=1e-8, shape=()):
        self.max = np.full(shape, -math.inf, np.float64)
        self.min = np.full(shape, math.inf, np.float64)
        self.epsilon = epsilon

    def update(self, x):
        batch_max = np.max(x, axis=0)
        batch_min = np.min(x, axis=0)

        if batch_max > self.max:
            self.max = batch_max
            
        if batch_min < self.min:
            self.min = batch_min

    def normalize(self, x, update=False):
        if update:
            self.update(x)
        return (x - self.min) / (self.max - self.min + self.epsilon)
        
    def __str__(self):
        return f"RangeNormalizer(max={self.max:.4f};min={self.min:.4f})"
