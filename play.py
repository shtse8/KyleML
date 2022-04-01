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
from games.Game import GameEventType, Player, Game

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


async def learning(game: Game, game_manager: GameManager):
    # network settings
    network = Network(np.prod(game.observation_shape), game.action_count)
    while True:
        scores = []
        steps = []

        # AI Play
        network.eval()
        for i in range(10):
            game.reset()
            while not game.is_done:
                for player in Player.get_players(game):
                    # game_manager.update()
                    try:
                        state_tensor = torch.tensor(np.array(player.state), dtype=torch.float).flatten().unsqueeze(0)
                        mask_tensor = torch.tensor(np.array(player.action_mask), dtype=torch.bool).unsqueeze(0)
                        probs = network(state_tensor, mask_tensor).squeeze(0)
                        # action = np.random.choice(len(probs), p=probs)
                        action = torch.argmax(probs)
                        reward = player.step(action)
                    except Exception as e:
                        print("Error:", e)
                    # finally:
                    #     await asyncio.sleep(0)
            scores += [player.score for player in Player.get_players(game)]

        print("Avg Score: " + str(np.average(scores)))
        # player plays first
        game.reset()
        while not game.is_done:
            game_manager.update()
            for event in game_manager.get_events():
                if event.event_type == GameEventType.Step:
                    try:
                        player = Player.get_players(game)[event.player_id]
                        state = player.state
                        mask = player.action_mask
                        action = event.value
                        reward = player.step(event.value)
                        steps += [(state, mask, action, reward)]
                    except Exception as e:
                        print("Error:", e)
            # await asyncio.sleep(1)

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


async def main():
    signal.signal(signal.SIGINT, signal_handler)
    if torch.cuda.is_available():
        print(f"CUDA {torch.version.cuda} (Devices: {torch.cuda.device_count()})")
    if torch.backends.cudnn.enabled:
        # torch.backends.cudnn.benchmark = True
        print(f"CUDNN {torch.backends.cudnn.version()}")

    game_manager = GameManager()
    game = game_manager.create("2048")
    asyncio.create_task(game_manager.render(game))
    asyncio.create_task(learning(game, game_manager))
    # print("bye")
    await asyncio.wait(asyncio.all_tasks())

if __name__ == "__main__":
    asyncio.run(main())
