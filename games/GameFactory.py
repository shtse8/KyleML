from .SimpleSnake import SimpleSnake
from .Snake import Snake
from .puzzle2048 import Puzzle2048
from .mario import Mario
from .gymgame import GymGame

class GameFactory:
    def __init__(self, name: str):
        self.name: str = name

    def get(self):
        try:
            if self.name == "snake":
                return Snake()
            if self.name == "simplesnake": 
                return SimpleSnake()
            if self.name == "cartpole":
               return GymGame("CartPole-v0")
            if self.name == "2048":
               return Puzzle2048()
            if self.name == "mario":
               return Mario()
            if self.name == "pong":
               return GymGame("Pong-v0")
            if self.name == "breakout":
               return GymGame("Breakout-v0")
            if self.name == "blackjack":
               return GymGame("Blackjack-v0")
        except ValueError:
            raise ValueError("Unknown Game " + self.name)