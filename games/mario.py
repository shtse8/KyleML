import numpy as np
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from .Game import Game
from .gymgame import GymGame

class Mario(GymGame):
    def __init__(self):
        super().__init__("SuperMarioBros-v0")
        self.game = JoypadSpace(self.game, COMPLEX_MOVEMENT)
        self.actionSpace = self.game.action_space.n
        print(self.actionSpace)