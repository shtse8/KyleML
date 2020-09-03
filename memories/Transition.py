class Transition(object):
    def __init__(self, state, hiddenStates, action: int, reward: float, nextState, nextHiddenStates, done: bool) -> None:
        self.state = state
        self.hiddenStates = hiddenStates
        self.action: int = action
        self.reward: float = reward
        self.nextState = nextState
        self.nextHiddenStates = nextHiddenStates
        self.done: bool = done
        self.value = 0
        self.advantage = 0
    # @property
    # def done(self) -> bool:
        # return self.nextState == None
