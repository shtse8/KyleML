class GameInfo:
    def __init__(self, state, mask, done, hidden_state=None):
        self.state = state
        self.mask = mask
        self.done = done
        self.hiddenState = hidden_state


class Game(object):
    def __init__(self):
        self.renderer = None
        self.reward = {}

    @property
    def action_spaces(self):
        raise NotImplementedError()

    @property
    def players(self):
        raise NotImplementedError()

    @property
    def action_count(self) -> int:
        return len(self.action_spaces)

    @property
    def player_count(self) -> int:
        return len(self.players)

    # Game methods
    def set_player(self, player_id: int):
        return GamePlayer(self, player_id)

    def get_player_count(self) -> int:
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()

    def is_done(self) -> bool:
        raise NotImplementedError()

    # Player Methods
    def can_step(self, player_id) -> bool:
        raise NotImplementedError()

    def get_info(self, player_id):
        return GameInfo(
            state=self.get_state(player_id),
            mask=self.get_mask(player_id),
            done=self.is_done())

    def get_mask(self, player_id: int):
        raise NotImplementedError()

    def get_state(self, player_id: int):
        raise NotImplementedError()

    def get_done_reward(self, player_id: int) -> float:
        return 0.

    def _step(self, player_id: int, action) -> None:
        raise NotImplementedError()

    def step(self, player_id: int, action) -> tuple:
        self._step(player_id, action)
        self.update()
        return self.get_reward(player_id)

    def get_reward(self, player_id) -> float:
        if player_id not in self.reward:
            return 0
        return self.reward[player_id]

    # UI Methods
    def render(self) -> None:
        pass

    def update(self) -> None:
        pass


class GamePlayer:
    def __init__(self, game: Game, player_id: int):
        self.game = game
        self.playerId = player_id

    def get_info(self):
        return self.game.get_info(self.playerId)

    def get_next(self):
        return GamePlayer(self.game, 1 + self.playerId % self.game.get_player_count())

    def get_state(self):
        return self.game.get_state(self.playerId)

    def can_step(self) -> bool:
        return self.game.can_step(self.playerId)

    def get_mask(self):
        return self.game.get_mask(self.playerId)

    def step(self, action) -> tuple:
        return self.game.step(self.playerId, action)

    def get_reward(self) -> float:
        return self.game.get_reward(self.playerId)

    def is_done(self) -> bool:
        return self.game.is_done()

    def get_done_reward(self) -> float:
        return self.game.get_done_reward(self.playerId)

