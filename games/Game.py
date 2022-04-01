from __future__ import annotations
from abc import abstractmethod, ABCMeta
from enum import Enum
from queue import SimpleQueue
from typing import Any

import pygame
from pygame.event import Event


class GameEventType(Enum):
    Step = 0


class GameEvent:
    def __init__(self, event_type: GameEventType, player_id: int, value: Any):
        self.event_type = event_type
        self.player_id = player_id
        self.value = value


class Game(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name
        self._core = None

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_spaces(self) -> list[any]:
        pass

    @property
    def action_count(self) -> int:
        return len(self.action_spaces)

    @property
    @abstractmethod
    def player_ids(self) -> list[int]:
        pass

    @property
    def player_count(self) -> int:
        return len(self.player_ids)

    @property
    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def _create_game_core(self):
        pass

    def reset(self):
        self._core = self._create_game_core()

    def update_display(self, surface: pygame.Surface):
        pass

    def process_event(self, event):
        pass

    @abstractmethod
    def can_step(self, player_id: int) -> bool:
        pass

    @abstractmethod
    def get_state(self, player_id: int):
        pass

    @abstractmethod
    def get_action_spaces_mask(self, player_id: int):
        pass

    @abstractmethod
    def get_score(self, player_id: int) -> float:
        pass

    # Return reward
    @abstractmethod
    def step(self, player_id: int, action) -> float:
        pass


class GetOnlySimpleQueue:
    def __init__(self, queue: SimpleQueue):
        self.__queue = queue

    def get(self):
        while not self.__queue.empty():
            yield self.__queue.get_nowait()


class RendererEvent(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Renderer:
    def __init__(self):
        self.__game: Game = None
        self.__event = SimpleQueue()
        self.width: int = 500
        self.height: int = 600
        pygame.init()

    @property
    def game(self) -> Game:
        return self.__game

    @game.setter
    def game(self, value: Game):
        self.__game = value
        # Update Caption
        pygame.display.set_caption(self.__game.name)

    @property
    def event(self) -> GetOnlySimpleQueue:
        return GetOnlySimpleQueue(self.__event)

    def update(self):
        if self.__game is None:
            return

        # Handler UI Events
        events: list[Event] = pygame.event.get()
        for event in events:
            try:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.__event.put(RendererEvent.UP)
                    if event.key == pygame.K_RIGHT:
                        self.__event.put(RendererEvent.RIGHT)
                    if event.key == pygame.K_DOWN:
                        self.__event.put(RendererEvent.DOWN)
                    if event.key == pygame.K_LEFT:
                        self.__event.put(RendererEvent.LEFT)
            except Exception as e:
                print(str(e))

        display = pygame.display.set_mode((self.width, self.height))
        self.__game.update_display(display)
        pygame.display.update(display.get_rect())


class Player:
    def __init__(self, game: Game, player_id: int):
        self.__game = game
        self.__player_id = player_id

    @property
    def can_step(self) -> bool:
        return self.__game.can_step(self.__player_id)

    @property
    def state(self):
        return self.__game.get_state(self.__player_id)

    @property
    def action_mask(self):
        return self.__game.get_action_spaces_mask(self.__player_id)

    @property
    def score(self) -> float:
        return self.__game.get_score(self.__player_id)

    # Return reward
    def step(self, action) -> float:
        return self.__game.step(self.__player_id, action)

    @staticmethod
    def get_players(game: Game) -> list[Player]:
        return [Player(game, i) for i in game.player_ids]
