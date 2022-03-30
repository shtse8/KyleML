import gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from .Game import Game
from .gymgame import GymGame


class Mario(GymGame):
    def __init__(self):
        super().__init__("SuperMarioBros-v0")
        self.game = JoypadSpace(self.game, COMPLEX_MOVEMENT)
        self.actionSpace = self.game.action_space.n
        print(self.actionSpace)