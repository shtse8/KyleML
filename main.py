import os
import sys
import signal
import argparse


from games.SimpleSnake import SimpleSnake
from games.Snake import Snake
from games.puzzle2048 import Puzzle2048
# from games.CartPole import CartPole
from games.CartPole import CartPole
from games.Breakout import Breakout
from games.mario import Mario
from games.Pong import Pong
# from agents.DQNAgent import DQNAgent
# from agents.A2CAgent import A2CAgent
# from agents.A3CAgent import A3CAgent
from agents.PPOAgent import Agent, PPOAlgo
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def signal_handler(sig, frame):
    print()
    print()
    print()
    print('You pressed Ctrl+C!')
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Kyle RL Playground')
    parser.add_argument('--game', default='2048', type=str)
    parser.add_argument('--agent', default='ppo', type=str)
    parser.add_argument('--render', action='store_true', dest="render")
    parser.add_argument('--delay', default=0, type=float)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train', default=True, action='store_true', dest="train")
    parser.add_argument('--eval', action='store_false', dest="train")
    args = parser.parse_args()
    print(args)
    signal.signal(signal.SIGINT, signal_handler)
    
    if torch.cuda.is_available():
        print(f"CUDA {torch.version.cuda}")
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        print(f"CUDNN {torch.backends.cudnn.version()}")

    # agents = {
        # "dqn": DQNAgent,
        # "a2c": A2CAgent,
        # "a3c": A3CAgent,
        # "ppo": PPOAgent
    # }

    games = {
        "snake": Snake,
        "simplesnake": SimpleSnake,
        "cartpole": CartPole,
        "2048": Puzzle2048,
        "mario": Mario,
        "pong": Pong,
        "breakout": Breakout,
    }

    if args.game in games:
        game = games[args.game]()
    else:
        raise ValueError("Unknown Game " + args.game)

    if args.render:
        game.render()

    # if args.agent in agents:
    #     agent = agents[args.agent](game)
    # else:
    #     raise ValueError("Unknown Agent " + args.agent)

    # if args.load or not args.train:
    #     agent.load()

    agent = Agent(PPOAlgo(), game)
    agent.run(train=args.train, delay=args.delay)


if __name__ == "__main__":
    main()
