import random
import numpy as np
import math

class Tree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.treeLenght = 2 * capacity - 1
        self.sumTree = np.zeros(self.treeLenght)
        self.minTree = np.full(self.treeLenght, math.inf)
        self.minTree2 = np.full(self.treeLenght, -math.inf)
        self.maxTree = np.full(self.treeLenght, -math.inf)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def update(self, tree_idx, p):
        # update leaf node
        change = p - self.sumTree[tree_idx]
        self.sumTree[tree_idx] = p
        self.minTree[tree_idx] = p
        self.minTree2[tree_idx] = p
        self.maxTree[tree_idx] = p
        
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = self.getParentIndex(tree_idx)
            self.sumTree[tree_idx] += change
            self.minTree[tree_idx] = min(self.minTree[self.getLeftIndex(tree_idx)], self.minTree[self.getRightIndex(tree_idx)])
            self.minTree2[tree_idx] = min(self.minTree2[self.getLeftIndex(tree_idx)], self.minTree2[self.getRightIndex(tree_idx)])
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
            if cl_idx >= self.treeLenght:        # reach bottom, end search
                break
            else:
                if self.minTree2[cl_idx] <= self.minTree2[idx]:
                    idx = cl_idx
                else:
                    idx = cr_idx
        return idx
        
    def add(self, p, data):
        tree_idx = self.getMinLeafIndex()
        data_idx = self.getDataIndex(tree_idx)
        self.data[data_idx] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        if self.n_entries < self.capacity:
            self.n_entries += 1
            
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
            if cl_idx >= self.treeLenght:        # reach bottom, end search
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.sumTree[cl_idx]:
                    idx = cl_idx
                else:
                    v -= self.sumTree[cl_idx]
                    idx = cr_idx

        return idx, self.sumTree[idx], self.data[self.getDataIndex(idx)]
# 10 + 10
# [0, 10] [10, 20]
tree = Tree(20)


# print(random.uniform(0, 55))
# tree.add(0.01, 0)
tree.add(1, 1)
tree.add(2, 2)
tree.add(3, 3)
tree.add(4, 4)
tree.add(5, 5)
tree.add(6, 6)
tree.add(7, 7)
tree.add(8, 8)
tree.add(9, 9)
tree.add(10, 10)


n = 5
segment = tree.sum() / n
for i in range(n):
    a = segment * i
    b = segment * (i + 1)
    print(a, b)
    s = a + random.random() * (b - a)
    print(s)
    
print(tree.get(0))
# for i in range(55):
    # print(i, tree.get(i))
print(tree.sumTree)
print(tree.maxTree)
print(tree.minTree)
print(tree.minTree2)