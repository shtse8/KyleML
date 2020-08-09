import random
import numpy as np
import collections

class Transition(object):
    def __init__(self, state, action: float, reward: float, nextState, done: bool) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.done = done
        
    # @property
    # def done(self) -> bool:
        # return self.nextState == None