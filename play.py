import math
import os
import warnings
import sys
import signal
import asyncio

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

    game_manager = GameManager("2048")
    player_game_container = game_manager.create()
    player_game_container.render()

    # network settings
    observation_size = np.prod(player_game_container.observation_shape)
    network = Network(np.prod(player_game_container.observation_shape), player_game_container.action_count)
    while True:

        # AI Play
        network.eval()
        scores = []
        for i in range(10):
            player_game_container.reset()
            while not player_game_container.is_done:
                player_game_container.update()
                try:
                    player = player_game_container.players[0]
                    state = player.state
                    mask = player.action_spaces_mask
                    flatten_state = list(np.array(state).flat)
                    state_tensor = torch.tensor([flatten_state], dtype=torch.float)
                    mask_tensor = torch.tensor([mask], dtype=torch.bool)
                    values = network(state_tensor, mask_tensor).squeeze(0).detach().numpy()
                    action = np.argmax(values)
                    player_game_container.players[0].step(action)
                except Exception as e:
                    print("Error: " + str(e))
            scores += [player_game_container.players[0].score]

        print("Avg Score: " + str(np.average(scores)))

        # player plays first
        steps = []
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
                        pass

        # AI learns from player steps
        print("Step collected: " + str(len(steps)))

        network.train()
        states = torch.tensor([list(np.array(x[0]).flat) for x in steps], dtype=torch.float).detach()
        masks = torch.tensor([x[1] for x in steps], dtype=torch.bool).detach()
        actions = torch.tensor([x[2] for x in steps], dtype=torch.int).detach()
        values = network(states, masks)
        # print(values)
        # test = values.gather(1, actions.unsqueeze(1)).squeeze(1)
        dist = torch.distributions.Categorical(probs=values)
        test = dist.log_prob(actions)
        test2 = torch.ones_like(test)
        loss = nn.MSELoss()(test, torch.ones_like(test))
        loss.backward()
        network.optimizer.step()




if __name__ == "__main__":
    asyncio.run(main())
