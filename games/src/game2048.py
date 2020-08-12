import numpy as np
import math

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
    def __init__(self, position, value = 2):
        super().__init__(position.x, position.y)
        self.value = value
        
        self.previousPosition = None
        self.mergedFrom       = None # Tracks tiles that merged together

    def savePosition(self):
        self.previousPosition = Coord(self.x, self.y)

    def updatePosition(self, position):
        self.x = position.x
        self.y = position.y

class Grid:
    def __init__(self, size, previousState = None):
        self.size = size
        self.cells =  self.fromState(previousState) if previousState != None else self.empty()
        
    def empty(self):
        return np.empty((self.size, self.size), dtype=object)
        
    def fromState(self, state):
        cells = []

        for i in range(self.size):
            row = cells[x] = []
            
            for y in range(self.size):
                tile = state[x][y]
                row.append(tile if Tile(tile.position, tile.value) else None)
                
        return cells
    
    def randomAvailableCell(self):
        cells = self.availableCells()

        if len(cells) > 0:
            return np.random.choice(cells)
            
            
    def availableCells(self):
        cells = []
        
        for x, y, tile in self.eachCell():
            if not tile:
                cells.append(Coord(x, y))
                
        return np.array(cells)

    def eachCell(self):
        for x in range(self.size):
            for y in range(self.size):
                yield x, y, self.cells[x][y]
            
    def cellsAvailable(self):
        return len(self.availableCells()) > 0
        
    def cellAvailable(self, cell):
        return not self.cellOccupied(cell)
        
    def cellOccupied(self, cell):
        return self.cellContent(cell) != None
        
    def cellContent(self, cell):
        if self.withinBounds(cell):
            return self.cells[cell.x][cell.y]
        else:
            return None
            
    def insertTile(self, tile):
        self.cells[tile.x][tile.y] = tile
        
    def removeTile(self, tile):
        self.cells[tile.x][tile.y] = None
        
    def withinBounds(self, position):
        return position.x >= 0 and position.x < self.size and position.y >= 0 and position.y < self.size
    
class game2048:

    def __init__(self, size):
        #create a new game and initialize two tiles
        self.startTiles = 2
        self.size = size
        self.grid = Grid(self.size)
        self.over        = False
        self.won         = False
        self.keepPlaying = True
        self.score       = 0
        # Add the initial tiles
        self.addStartTiles()
    
    def addStartTiles(self):
        for i in range(self.startTiles):
            self.addRandomTile()
            
    def addRandomTile(self):
        if self.grid.cellsAvailable():
            value = 2 if np.random.uniform() < 0.9 else 4
            tile = Tile(self.grid.randomAvailableCell(), value)
            
            self.grid.insertTile(tile)
            
    def prepareTiles(self):
        for _, _, tile in self.grid.eachCell():
            if tile:
                tile.mergedFrom = None
                tile.savePosition()
                
    def moveTile(self, tile, cell):
        self.grid.cells[tile.x][tile.y] = None
        self.grid.cells[cell.x][cell.y] = tile
        tile.updatePosition(cell)
        
    def isGameTerminated(self):
        return self.over or (self.won and not self.keepPlaying)
        
    def move(self, direction):
        if self.isGameTerminated():
            return # Don't do anything if the game's over

        vector = self.getVector(direction)
        traversals = self.buildTraversals(vector)
        moved = False

        self.prepareTiles()

        # Traverse the grid in the right direction and move tiles
        for x in traversals.x:
            for y in traversals.y:
                cell = Coord(x, y)
                tile = self.grid.cellContent(cell)
                if tile:
                    positions = self.findFarthestPosition(cell, vector)
                    next = self.grid.cellContent(positions["next"])
                        
                    # Only one merger per row traversal?
                    if next and next.value == tile.value and not next.mergedFrom:
                        merged = Tile(positions["next"], tile.value * 2)
                        merged.mergedFrom = [tile, next]
                        
                        self.grid.insertTile(merged)
                        self.grid.removeTile(tile)
                        
                        # Converge the two tiles' positions
                        tile.updatePosition(positions["next"])

                        # Update the score
                        self.score += merged.value

                        # The mighty 2048 tile
                        if merged.value == 2048:
                            self.won = True
                            
                    else:
                        self.moveTile(tile, positions["farthest"])
                    

                    if not self.positionsEqual(cell, tile):
                        moved = True # The tile moved from its original cell!
                        
        if moved:
            self.addRandomTile()

            if not self.movesAvailable():
                self.over = True # Game over!
       
    def getVector(self, direction):
        # Coords representing tile movement
        map = [
            Vector(0, -1), # Up
            Vector(1, 0),  # Right
            Vector(0, 1),  # Down
            Vector(-1, 0)  # Left
        ]

        return map[direction]
            
    def buildTraversals(self, vector):
        traversals = Traversal();
        
        for pos in range(self.size):
            traversals.x.append(pos)
            traversals.y.append(pos)
            
        # Always traverse from the farthest cell in the chosen direction
        if vector.x == 1:
            traversals.x.reverse()
        if vector.y == 1:
            traversals.y.reverse()
            
        return traversals

    def findFarthestPosition(self, cell, vector):
        # Progress towards the vector direction until an obstacle is found
        while True:
            previous = cell;
            cell     = Coord(previous.x + vector.x, previous.y + vector.y)
            if not (self.grid.withinBounds(cell) and self.grid.cellAvailable(cell)):
                break
            

        return {
            "farthest": previous,
            "next": cell # Used to check if a merge is required
        }
    
    def movesAvailable(self):
        return self.grid.cellsAvailable() or self.tileMatchesAvailable()
        
    def tileMatchesAvailable(self):
        for x in range(self.size):
            for y in range(self.size):
                tile = self.grid.cellContent(Coord(x, y))
                
                if tile:
                    for direction in range(4):
                        vector = self.getVector(direction)
                        cell = Coord(x + vector.x, y + vector.y)
                        
                        other = self.grid.cellContent(cell)
                        
                        if other and other.value == tile.value:
                            return True
        return False
        
    def positionsEqual(self, first, second):
        return first.x == second.x and first.y == second.y
        
    
