import math
import os
import warnings
import sys
import signal
import asyncio
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from games.Game import GameEventType

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
warnings.filterwarnings("ignore", category=DeprecationWarning)

from games.GameManager import GameManager


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def signal_handler(sig, frame):
    print()
    print()
    print()
    print('You pressed Ctrl+C!')
    sys.exit(0)


class Network(nn.Module):

    def __init__(self, in_nodes: int, out_nodes: int):
        super(Network, self).__init__()
        hidden_nodes = 256
        self.layers = nn.Sequential(
            nn.Linear(in_nodes, hidden_nodes),
            nn.ELU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ELU(),
            nn.Linear(hidden_nodes, out_nodes)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x, m):
        x = self.layers(x)
        if m is not None:
            x = x.masked_fill(~m, -math.inf)
        x = F.softmax(x, dim=-1)
        return x


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    if torch.cuda.is_available():
        print(f"CUDA {torch.version.cuda} (Devices: {torch.cuda.device_count()})")
    if torch.backends.cudnn.enabled:
        # torch.backends.cudnn.benchmark = True
        print(f"CUDNN {torch.backends.cudnn.version()}")

    game_manager = GameManager("2048")
    player_game_container = game_manager.create()
    player_game_container.render()

    # network settings
    network = Network(np.prod(player_game_container.observation_shape), player_game_container.action_count)
    while True:
        scores = []
        steps = []

        # AI Play
        network.eval()
        for i in range(10):
            player_game_container.reset()
            while not player_game_container.is_done:
                player_game_container.update()
                try:
                    player = player_game_container.players[0]
                    state = player.state
                    mask = player.action_spaces_mask
                    flatten_state = list(np.array(state).flat)
                    state_tensor = torch.tensor(np.array([flatten_state]), dtype=torch.float)
                    mask_tensor = torch.tensor(np.array([mask]), dtype=torch.bool)
                    probs = network(state_tensor, mask_tensor).squeeze(0).detach().numpy()
                    # action = np.random.choice(len(probs), p=probs)
                    action = np.argmax(probs)
                    reward = player_game_container.players[0].step(action)
                    # if i == 0:
                    #     time.sleep(0.1)
                except Exception as e:
                    print("Error: " + str(e))
            scores += [player_game_container.players[0].score]

        print("Avg Score: " + str(np.average(scores)))
        # player plays first
        player_game_container.reset()
        while not player_game_container.is_done:
            player_game_container.update()
            for event in player_game_container.get_events():
                if event.event_type == GameEventType.Step:
                    try:
                        player = player_game_container.players[0]
                        state = player.state
                        mask = player.action_spaces_mask
                        action = event.value
                        reward = player_game_container.players[event.player_id].step(event.value)
                        steps += [(state, mask, action, reward)]
                    except Exception as e:
                        print("Error:", e)

        # AI learns from player steps
        print("Step collected: " + str(len(steps)))
        if len(steps) > 0:
            network.train()
            states = torch.tensor(np.array([list(np.array(x[0]).flat) for x in steps]), dtype=torch.float).detach()
            masks = torch.tensor(np.array([x[1] for x in steps]), dtype=torch.bool).detach()
            actions = torch.tensor(np.array([x[2] for x in steps]), dtype=torch.int).detach()

            probs = network(states, masks)

            dist = torch.distributions.Categorical(probs=probs)
            prob = dist.log_prob(actions)
            reward = torch.ones_like(prob)
            loss = nn.MSELoss()(prob, reward)
            print("loss =", loss.item())
            loss.backward()
            network.optimizer.step()


if __name__ == "__main__":
    asyncio.run(main())
