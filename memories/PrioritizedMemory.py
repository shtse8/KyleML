import random
import numpy as np
import math 

class PrioritizedMemory(object):
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = Tree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, sample):
        # print(self.tree.max(), self.tree.min(), self.tree.sum())
        max_p = self.tree.max()
        if max_p == -math.inf:
            max_p = 1
        # print(max_p)
        self.tree.add(max_p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        
        # proportional prioritization
        # segmented for preventing duplicated item
        segment = self.tree.sum() / n 
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            # s = random.uniform(a, b)
            # We shouldn't use uniform because the margins will be overlapped.
            s = random.random() * ( b - a ) + a
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # = (1 / N * 1 / P(i)) ^ beta = (N * P(i)) ^ -beta
        sampling_probabilities = priorities / self.tree.sum()
        is_weight = np.power(self.tree.entries * sampling_probabilities, -self.beta)
        
        # Normalize to [0, 1]
        min_probabilities = self.tree.min() / self.tree.sum()
        max_is_weight = np.power(self.tree.entries * min_probabilities, -self.beta)
        
        is_weight /= max_is_weight

        return idxs, batch, is_weight

    def batch_update(self, tree_idx, abs_errors):
        for ti, e in zip(tree_idx, abs_errors):
            p = self._get_priority(e)
            self.tree.update(ti, p)



class Tree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.treeLength = 2 * capacity - 1
        self.sumTree = np.zeros(self.treeLength)
        self.minTree = np.full(self.treeLength, math.inf)
        #self.minTree2 = np.full(self.treeLenght, -math.inf)
        self.maxTree = np.full(self.treeLength, -math.inf)
        self.data = np.zeros(capacity, dtype=object)
        self.entries = 0

    def update(self, tree_idx, p):
        # update leaf node
        
        change = p - self.sumTree[tree_idx]
        self.sumTree[tree_idx] = p
        self.minTree[tree_idx] = p
        self.maxTree[tree_idx] = p
        
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = self.getParentIndex(tree_idx)
            self.sumTree[tree_idx] += change
            self.minTree[tree_idx] = min(self.minTree[self.getLeftIndex(tree_idx)], self.minTree[self.getRightIndex(tree_idx)])
            self.maxTree[tree_idx] = max(self.maxTree[self.getLeftIndex(tree_idx)], self.maxTree[self.getRightIndex(tree_idx)])
            
    def sum(self):
        return self.sumTree[0]
 
    def min(self):
        return self.minTree[0]
 
    def max(self):
        return self.maxTree[0]

    def getMinLeafIndex(self):
        idx = 0
        while True:
            cl_idx = self.getLeftIndex(idx)        # this leaf's left and right kids
            cr_idx = self.getRightIndex(idx)
            if cl_idx >= self.treeLength:        # reach bottom, end search
                break
            else:
                if self.minTree[cl_idx] <= self.minTree[idx]:
                    idx = cl_idx
                else:
                    idx = cr_idx
        return idx
        
    def add(self, p, data):
        if self.entries < self.capacity:
            tree_idx = self.capacity + self.entries - 1
        else:
            tree_idx = self.getMinLeafIndex()
        data_idx = self.getDataIndex(tree_idx)
        self.data[data_idx] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        if self.entries < self.capacity:
            self.entries += 1
            
    def getParentIndex(self, tree_idx):
        return (tree_idx - 1) // 2
    
    def getLeftIndex(self, tree_idx):
        return 2 * tree_idx + 1
        
    def getRightIndex(self, tree_idx):
        return self.getLeftIndex(tree_idx) + 1
        
    def getDataIndex(self, tree_idx):
        return tree_idx + 1 - self.capacity
        
    def get(self, v):
        idx = 0
        while True:
            cl_idx = self.getLeftIndex(idx)
            cr_idx = self.getRightIndex(idx)
            if cl_idx >= self.treeLength:        # reach bottom, end search
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.sumTree[cl_idx]:
                    idx = cl_idx
                else:
                    v -= self.sumTree[cl_idx]
                    idx = cr_idx

        return idx, self.sumTree[idx], self.data[self.getDataIndex(idx)]

