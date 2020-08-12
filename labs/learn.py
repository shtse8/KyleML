import random
import numpy as np


class Tree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.treeLenght = 2 * capacity - 1
        self.sumTree = np.zeros(self.treeLenght)
        self.minTree = np.zeros(self.treeLenght)
        self.maxTree = np.zeros(self.treeLenght)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

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
            m = max(self.maxTree[self.getLeftIndex(tree_idx)], self.maxTree[self.getRightIndex(tree_idx)])
            print("max", m, "sum", self.sumTree[tree_idx])
            self.maxTree[tree_idx] = m
            
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
                if self.minTree[cl_idx] <= self.minTree[idx]:
                    idx = cl_idx
                else:
                    idx = cr_idx
        return idx
        
    def add(self, p, data):
        tree_idx = self.getMinLeafIndex()
        data_pointer = tree_idx + 1 - self.capacity
        self.data[data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def getParentIndex(self, idx):
        return (idx - 1) // 2
    
    def getLeftIndex(self, idx):
        return 2 * idx + 1
        
    def getRightIndex(self, idx):
        return self.getLeftIndex(idx) + 1
        
    def get(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = self.getLeftIndex(parent_idx)        # this leaf's left and right kids
            cr_idx = self.getRightIndex(parent_idx)
            if cl_idx >= self.treeLenght:        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.sumTree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.sumTree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.sumTree[leaf_idx], self.data[data_idx]


print(np.random.random_integers(0, 5, 5))
tree = Tree(5)

tree.add(1, 1)
tree.add(2, 2)
tree.add(3, 3)
tree.add(4, 4)
tree.add(5, 5)
tree.add(6, 6)
tree.add(7, 7)

print(tree.get(100))
print(tree.sumTree)
print(tree.maxTree)
print(tree.minTree)