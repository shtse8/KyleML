from .gymgame import GymGame


class Pong(GymGame):
    def __init__(self):
        super().__init__("Pong-v0")
    