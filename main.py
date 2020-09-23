import os
import sys
import signal
import argparse
import asyncio


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
        print(f"CUDA {torch.version.cuda} (Devices: {torch.cuda.device_count()})")
    if torch.backends.cudnn.enabled:
        # torch.backends.cudnn.benchmark = True
        print(f"CUDNN {torch.backends.cudnn.version()}")

    # agents = {
        # "dqn": DQNAgent,
        # "a2c": A2CAgent,
        # "a3c": A3CAgent,
        # "ppo": PPOAgent
    # }

    gameFactory = GameFactory(args.game)
    
    # if args.agent in agents:
    #     agent = agents[args.agent](game)
    # else:
    #     raise ValueError("Unknown Agent " + args.agent)

    # if args.load or not args.train:
    #     agent.load()

    rl = RL(KyleAlgo(), gameFactory)
    await rl.run(train=args.train, load=args.load, delay=args.delay)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
