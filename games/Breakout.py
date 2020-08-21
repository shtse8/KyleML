from .gymgame import GymGame


class Breakout(GymGame):
    def __init__(self):
        super().__init__("Breakout-v0")
    