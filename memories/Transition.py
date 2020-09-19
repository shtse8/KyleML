class Transition(object):
    def __init__(self, state, action: int, reward: float, nextState, done: bool) -> None:
        self.state = state
        self.hiddenState = None
        self.action: int = action
        self.reward: float = reward
        self.nextState = nextState
        self.nextHiddenState = None
        self.done: bool = done
        self.value = 0
        self.advantage = 0
    # @property
    # def done(self) -> bool:
        # return self.nextState == None
