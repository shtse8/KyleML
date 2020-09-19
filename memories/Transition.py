class Transition(object):
    def __init__(self) -> None:
        self.info = None
        self.action: int = None
        self.reward: float = None
        self.next = None
