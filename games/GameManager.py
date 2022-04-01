import asyncio
from typing import Iterator

from utils.Event import Event
from .SimpleSnake import SimpleSnake
from .Snake import Snake
from .connect4 import Connect4
from .gomoku import Gomoku
# from .mario import Mario
from .gymgame import GymGame
from .puzzle2048 import Puzzle2048
from .tictactoe import TicTacToe
from .Game import Game, Renderer, GetOnlySimpleQueue, GameEvent


class GameManager:
    def __init__(self):
        self.__renderer: Renderer = None
        self.__games = []

    def create(self, name: str) -> Game:
        if name == "snake":
            return Snake()
        elif name == "simple-snake":
            return SimpleSnake()
        elif name == "cartpole":
            return GymGame("CartPole-v0")
        elif name == "2048":
            return Puzzle2048()
        # elif self.name == "mario":
        #     self._game = Mario()
        elif name == "pong":
            return GymGame("Pong-v0")
        elif name == "breakout":
            return GymGame("Breakout-v0")
        elif name == "blackjack":
            return GymGame("Blackjack-v0")
        elif name == "pacman":
            return GymGame("MsPacman-v0")
        elif name == "simple-pacman":
            return GymGame("MsPacman-ram-v0")
        elif name == "pinball":
            return GymGame("VideoPinball-v0")
        elif name == "simple-pinball":
            return GymGame("VideoPinball-ram-v0")
        elif name == "tictactoe":
            return TicTacToe()
        elif name == "gomoku":
            return Gomoku()
        elif name == "connect4":
            return Connect4()
        else:
            raise ValueError("Unknown Game " + name)

    async def render(self, game: Game):
        if self.__renderer is not None:
            return

        self.__renderer = Renderer()
        self.__renderer.game = game

        # while True:
        #     self.__renderer.update()
        #     await asyncio.sleep(1 / 10)

    def update(self):
        if self.__renderer is None:
            return
        self.__renderer.update()

    def get_events(self) -> Iterator[GameEvent]:
        if self.__renderer is None:
            return
        for event in self.__renderer.event.get():
            yield self.__renderer.game.process_event(event)
