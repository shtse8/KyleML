import numpy as np


class Traversal:
    def __init__(self):
        self.x = []
        self.y = []


class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Vector(Coord):
    def __init__(self, x, y):
        super().__init__(x, y)


class Tile(Coord):
    def __init__(self, position, value=2):
        super().__init__(position.x, position.y)
        self.value = value

        self.previousPosition = None
        self.mergedFrom = None  # Track tiles that merged together

    def save_position(self):
        self.previousPosition = Coord(self.x, self.y)

    def update_position(self, position):
        self.x = position.x
        self.y = position.y


class Grid:
    def __init__(self, size, previous_state=None):
        self.size = size
        self.cells = self.from_state(previous_state) if previous_state is not None else self.empty()

    def empty(self):
        return np.empty((self.size, self.size), dtype=object)

    def from_state(self, state):
        cells = []

        for i in range(self.size):
            row = cells[x] = []

            for y in range(self.size):
                tile = state[x][y]
                row.append(tile if Tile(tile.position, tile.value) else None)

        return cells

    def available_cells(self):
        cells = []

        for x, y, tile in self.each_cell():
            if not tile:
                cells.append(Coord(x, y))

        return np.array(cells)

    def each_cell(self):
        for x in range(self.size):
            for y in range(self.size):
                yield x, y, self.cells[x][y]

    def cells_available(self):
        return len(self.available_cells()) > 0

    def cell_available(self, cell):
        return not self.cell_occupied(cell)

    def cell_occupied(self, cell):
        return self.cell_content(cell) is not None

    def cell_content(self, cell):
        if self.within_bounds(cell):
            return self.cells[cell.x][cell.y]
        else:
            return None

    def insert_tile(self, tile):
        self.cells[tile.x][tile.y] = tile

    def remove_tile(self, tile):
        self.cells[tile.x][tile.y] = None

    def within_bounds(self, position):
        return 0 <= position.x < self.size and 0 <= position.y < self.size


class Game2048:

    def __init__(self, size, seed=None):
        # create a new game and initialize two tiles
        self.startTiles = 2
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.grid = Grid(self.size)
        self.over = False
        self.won = False
        self.keepPlaying = True
        self.score = 0
        # Add the initial tiles
        self.add_start_tiles()

    def add_start_tiles(self):
        for i in range(self.startTiles):
            self.add_random_tile()

    def add_random_tile(self):
        if self.grid.cells_available():
            value = 2 if self.rng.uniform() < 0.9 else 4

            cells = self.grid.available_cells()
            tile = Tile(self.rng.choice(cells), value)

            self.grid.insert_tile(tile)

    def prepare_tiles(self):
        for _, _, tile in self.grid.each_cell():
            if tile:
                tile.mergedFrom = None
                tile.save_position()

    def move_tile(self, tile, cell):
        self.grid.cells[tile.x][tile.y] = None
        self.grid.cells[cell.x][cell.y] = tile
        tile.update_position(cell)

    def is_game_terminated(self):
        return self.over or (self.won and not self.keepPlaying)

    def move(self, direction: int):
        if self.is_game_terminated():
            raise Exception("Game is over.")  # Don't do anything if the game's over

        vector = self.get_vector(direction)
        traversals = self.build_traversals(vector)
        moved = False

        self.prepare_tiles()
        # Traverse the grid in the right direction and move tiles
        for x in traversals.x:
            for y in traversals.y:
                cell = Coord(x, y)
                tile = self.grid.cell_content(cell)
                if tile:
                    positions = self.find_farthest_position(cell, vector)
                    next_cell = self.grid.cell_content(positions["next"])

                    # Only one merger per row traversal?
                    if next_cell and next_cell.value == tile.value and not next_cell.mergedFrom:
                        merged = Tile(positions["next"], tile.value * 2)
                        merged.mergedFrom = [tile, next_cell]

                        self.grid.insert_tile(merged)
                        self.grid.remove_tile(tile)

                        # Converge the two tiles' positions
                        tile.update_position(positions["next"])

                        # Update the score
                        self.score += merged.value

                        # The mighty 2048 tile
                        if merged.value == 2048:
                            self.won = True

                    else:
                        self.move_tile(tile, positions["farthest"])

                    if not self.positions_equal(cell, tile):
                        moved = True  # The tile moved from its original cell!

        if moved:
            self.add_random_tile()

            if not self.moves_available():
                self.over = True  # Game over!
        else:
            raise Exception("Invalid move")

    @staticmethod
    def get_vector(direction: int):
        # Coords representing tile movement
        vector = [
            Vector(0, -1),  # Up
            Vector(1, 0),  # Right
            Vector(0, 1),  # Down
            Vector(-1, 0)  # Left
        ]

        return vector[direction]

    def build_traversals(self, vector):
        traversals = Traversal()

        for pos in range(self.size):
            traversals.x.append(pos)
            traversals.y.append(pos)

        # Always traverse from the farthest cell in the chosen direction
        if vector.x == 1:
            traversals.x.reverse()
        if vector.y == 1:
            traversals.y.reverse()

        return traversals

    def find_farthest_position(self, cell, vector):
        # Progress towards the vector direction until an obstacle is found
        while True:
            previous = cell
            cell = Coord(previous.x + vector.x, previous.y + vector.y)
            if not (self.grid.within_bounds(cell) and self.grid.cell_available(cell)):
                break

        return {
            "farthest": previous,
            "next": cell  # Used to check if a merge is required
        }

    def moves_available(self):
        return self.grid.cells_available() or self.tile_matches_available()

    def tile_matches_available(self):
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cell_content(Coord(x, y))

                if tile:
                    for direction in range(4):
                        vector = self.get_vector(direction)
                        cell = Coord(x + vector.x, y + vector.y)

                        other = self.grid.cell_content(cell)

                        if other and other.value == tile.value:
                            return True
        return False

    @staticmethod
    def positions_equal(first, second):
        return first.x == second.x and first.y == second.y
