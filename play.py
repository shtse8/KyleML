import os
import warnings
import sys
import signal
import asyncio

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


async def main():
    signal.signal(signal.SIGINT, signal_handler)

    game_manager = GameManager("2048")
    player_game_container = game_manager.create()
    player_game_container.render()
    while True:
        # player plays first
        player_game_container.reset()
        while not player_game_container.is_done:
            player_game_container.update()
            for event in player_game_container.get_events():
                if event.event_type == GameEventType.Step:
                    player_game_container.players[event.player_id].step(event.value)

        # AI learns from player steps

        # test AI


if __name__ == "__main__":
    asyncio.run(main())
