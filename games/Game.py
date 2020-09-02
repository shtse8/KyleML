class Game(object):
    def __init__(self):
        self.rendered = False

    def getPlayerCount(self):
        raise NotImplementedError()

    def canStep(self, playerId):
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()

    def getMask(self):
        raise NotImplementedError()

    def getState(self):
        raise NotImplementedError()

    def _step(self, action) -> None:
        raise NotImplementedError()

    def step(self, action) -> tuple:
        self._step(action)
        self.update()
        return self.getState(), self.getReward(), self.isDone()

    def getReward(self) -> float:
        return 0

    def isDone(self) -> bool:
        raise NotImplementedError()

    def render(self) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        raise NotImplementedError()

