import random
import numpy as np
import collections

def unpack(traces):
    """Returns states, actions, rewards, end_states, and a mask for episode boundaries given traces."""
    states = [t[0].state for t in traces]
    actions = [t[0].action for t in traces]
    rewards = [[e.reward for e in t] for t in traces]
    end_states = [t[-1].nextState for t in traces]
    not_done_mask = [[1 if n.nextState is not None else 0 for n in t] for t in traces]
    print(states, actions, rewards, end_states, not_done_mask)
    return states, actions, rewards, end_states, not_done_mask

class OnPolicy:
    """Stores multiple steps of interaction with multiple environments."""
    def __init__(self, steps=1, instances=1):
        self.buffers = [[] for _ in range(instances)]
        self.steps = steps
        self.instances = instances

    def put(self, transition, instance=0):
        """Stores transition into the appropriate buffer."""
        self.buffers[instance].append(transition)

    def get(self):
        """Returns all traces and clears the memory."""
        traces = [list(tb) for tb in self.buffers]
        self.buffers = [[] for _ in range(self.instances)]
        return unpack(traces)

    def __len__(self):
        """Returns the number of traces stored."""
        return sum([len(b) - self.steps + 1 for b in self.buffers])