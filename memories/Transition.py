class Transition(object):
    def __init__(self, state, action: int, reward: float, nextState, done: bool, next_hiddenStates) -> None:
        self.state = state
        self.action: int = action
        self.reward: float = reward
        self.nextState = nextState
        self.next_hiddenStates = next_hiddenStates
        self.done: bool = done
        self.value = 0
        self.advantage = 0
    # @property
    # def done(self) -> bool:
        # return self.nextState == None
