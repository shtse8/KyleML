import collections
import itertools
import numpy as np
import time

from memories.Transition import Transition


class SimpleMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.current = 0

    def add(self, transition: Transition) -> None:
        self.memory.append(transition)

    def get(self, size: int = 0, start: int = 0):
        if size > 0:
            end = min(start + size, len(self.memory) - 1)
            return list(itertools.islice(self.memory, start, end))
            # return np.array(self.memory)[start:end]
        else:
            return self.memory
    
    def clear(self) -> None:
        self.memory.clear()
    
    def getLast(self, num: int):
        return np.array(self.memory)[-num:]

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, i: int):
        return self.memory[i]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < len(self.memory):
            result = self.memory[self.current]
            self.current += 1
            return result
        self.current = 0
        raise StopIteration
