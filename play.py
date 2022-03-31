import os
import warnings
import sys
import signal
import argparse
import asyncio

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
warnings.filterwarnings("ignore", category=DeprecationWarning)

# from agents.DQNAgent import DQNAgent
# from agents.A2CAgent import A2CAgent
# from agents.A3CAgent import A3CAgent
from games.GameFactory import GameFactory
from agents.Agent import RL
from agents.PPOAgent import PPOAlgo
from agents.KyleAgent import KyleAlgo
from agents.AlphaZeroAgent import AlphaZeroAlgo
import torch


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def signal_handler(sig, frame):
    print()
    print()
    print()
    print('You pressed Ctrl+C!')
    sys.exit(0)


async def main():
    signal.signal(signal.SIGINT, signal_handler)

    game_factory = GameFactory("2048")
    game = game_factory.get()
    render = game.render()
    game.reset()
    while not game.is_done():
        render.update()


if __name__ == "__main__":
    asyncio.run(main())
