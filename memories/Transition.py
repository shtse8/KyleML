class Transition(object):
    def __init__(self, state, action: int, reward: float, nextState, done: bool) -> None:
        self.state = state
        self.action: int = action
        self.reward: float = reward
        self.nextState = nextState
        self.done: bool = done

    # @property
    # def done(self) -> bool:
        # return self.nextState == None
