import os
import sys
import signal
import argparse


from games.SimpleSnake import SimpleSnake
from games.Snake import Snake
from games.puzzle2048 import Puzzle2048
# from games.CartPole import CartPole
from games.CartPole import CartPole
from agents.DQNAgent import DQNAgent
from agents.A2CAgent import A2CAgent
from agents.A3CAgent import A3CAgent
from agents.PPOAgent import PPOAgent
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--train', default=True, action='store_true', dest="train")
    parser.add_argument('--play', action='store_false', dest="train")
    args = parser.parse_args()
    print(args)
    signal.signal(signal.SIGINT, signal_handler)
    print("CUDA:", torch.cuda.is_available())
    if args.game == "2048":
        game = Puzzle2048()
    elif args.game == "snake":
        game = Snake()
    elif args.game == "simplesnake":
        game = SimpleSnake()
    elif args.game == "cartpole":
        game = CartPole()
    else:
        raise ValueError("Unknown Game " + args.game)

    if args.render:
        game.render()

    if args.agent == "a3c":
        agent = A3CAgent(game)
    elif args.agent == "a2c":
        agent = A2CAgent(game)
    elif args.agent == "dqn":
        agent = DQNAgent(game)
    elif args.agent == "ppo":
        agent = PPOAgent(game)
    else:
        raise ValueError("Unknown Agent " + args.agent)

    if args.load or not args.train:
        agent.load()

    agent.run(train=args.train)


if __name__ == "__main__":
    main()
