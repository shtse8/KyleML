import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import signal
# from games.Snake import Snake
from games.puzzle2048 import Puzzle2048
from agents.DQNAgent import DQNAgent
from agents.A2CAgent import A2CAgent

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    env = Puzzle2048()
    agent = A2CAgent(env)
    # agent.load()
    # agent.printSummary()
    agent.train()
    # agent.play(render=True)
    
            
if __name__ == "__main__":
    main()