class Transition(object):
    def __init__(self, state, hiddenState, action: int, reward: float, nextState, nextHiddenState, done: bool) -> None:
        self.state = state
        self.hiddenState = hiddenState
        self.action: int = action
        self.reward: float = reward
        self.nextState = nextState
        self.nextHiddenState = nextHiddenState
        self.done: bool = done
        self.value = 0
        self.advantage = 0
    # @property
    # def done(self) -> bool:
        # return self.nextState == None
