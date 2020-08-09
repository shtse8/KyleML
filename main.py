import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import signal
from games.Snake import Snake
from agents.DQN.agent import Agent

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    env = Snake()
    agent = Agent(env)
    # agent.load()
    agent.printSummary()
    agent.train()
    # agent.play(render=True)
    
            
if __name__ == "__main__":
    main()