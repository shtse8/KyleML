from .gymgame import GymGame


class CartPole(GymGame):
    def __init__(self):
        super().__init__("CartPole-v0")
    