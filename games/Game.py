class GameInfo:
    def __init__(self, state, mask, done, hiddenState=None):
        self.state = state
        self.mask = mask
        self.done = done
        self.hiddenState = hiddenState


class Game(object):
    def __init__(self):
        self.renderer = None
        self.reward = {}

    @property
    def actionSpaces(self):
        raise NotImplementedError()

    @property
    def players(self):
        raise NotImplementedError()

    @property
    def actionCount(self):
        return len(self.actionSpaces)

    @property
    def playerCount(self):
        return len(self.players)

    # Game methods
    def setPlayer(self, playerId: int):
        return GamePlayer(self, playerId)

    def getPlayerCount(self):
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()

    def isDone(self) -> bool:
        raise NotImplementedError()

    # Player Methods
    def canStep(self, playerId):
        raise NotImplementedError()

    def getInfo(self, playerId):
        return GameInfo(
            state=self.getState(playerId),
            mask=self.getMask(playerId),
            done=self.isDone())

    def getMask(self, playerId: int):
        raise NotImplementedError()

    def getState(self, playerId: int):
        raise NotImplementedError()

    def getDoneReward(self, playerId: int) -> float:
        return 0.

    def _step(self, playerId: int, action) -> None:
        raise NotImplementedError()

    def step(self, playerId: int, action) -> tuple:
        self._step(playerId, action)
        self.update()
        return self.getReward(playerId)

    def getReward(self, playerId) -> float:
        if playerId not in self.reward:
            return 0
        return self.reward[playerId]

    # UI Methods
    def render(self) -> None:
        pass

    def update(self) -> None:
        pass


class GamePlayer:
    def __init__(self, game: Game, playerId: int):
        self.game = game
        self.playerId = playerId

    def getInfo(self):
        return self.game.getInfo(self.playerId)

    def getNext(self):
        return GamePlayer(self.game, 1 + self.playerId % self.game.getPlayerCount())

    def getState(self):
        return self.game.getState(self.playerId)

    def canStep(self):
        return self.game.canStep(self.playerId)

    def getMask(self):
        return self.game.getMask(self.playerId)

    def step(self, action) -> tuple:
        return self.game.step(self.playerId, action)

    def getReward(self) -> float:
        return self.game.getReward(self.playerId)

    def isDone(self) -> bool:
        return self.game.isDone()

    def getDoneReward(self) -> float:
        return self.game.getDoneReward(self.playerId)

