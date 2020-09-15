from .SimpleSnake import SimpleSnake
from .Snake import Snake
from .puzzle2048 import Puzzle2048
from .mario import Mario
from .gymgame import GymGame
from .tictactoe import TicTacToe
from .gomoku import Gomoku
from .connect4 import Connect4

class GameFactory:
    def __init__(self, name: str):
        self.name: str = name

    def get(self):
        if self.name == "snake":
            return Snake()
        if self.name == "simple-snake": 
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
        if self.name == "pacman":
            return GymGame("MsPacman-v0")
        if self.name == "simple-pacman":
            return GymGame("MsPacman-ram-v0")
        if self.name == "pinball":
            return GymGame("VideoPinball-v0")
        if self.name == "simple-pinball":
            return GymGame("VideoPinball-ram-v0")
        if self.name == "tictactoe":
            return TicTacToe()
        if self.name == "gomoku":
            return Gomoku()
        if self.name == "connect4":
            return Connect4()
        raise ValueError("Unknown Game " + self.name)