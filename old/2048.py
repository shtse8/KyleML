# -*- coding: utf-8 -*-
# Rundong Li, UESTC
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame, sys, random
from pygame.locals import *
from enum import Enum
from DQN import Agent
import numpy as np
import math

BOARD_WIDTH = 4
BOARD_HEIGHT = 4
BOARD_OUTER_LINE_WIDTH = 4
BLOCK_SIZE = 100
MARGIN_SIZE = 20
TITLE_SIZE = 72
FONT_SIZE = 64
RESULT_SIZE = 50
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 640
MAX_SCORE = 2048
INIT_SCORE = 2
FPS = 30

# check window size
assert (BOARD_WIDTH * (BLOCK_SIZE + MARGIN_SIZE) < WINDOW_WIDTH) and \
       (BOARD_HEIGHT * (BLOCK_SIZE + MARGIN_SIZE) + TITLE_SIZE < WINDOW_HEIGHT), \
       'window_size is much too small!'
X_MARGIN = int((WINDOW_WIDTH - (BOARD_WIDTH * (BLOCK_SIZE + MARGIN_SIZE))) / 2)
Y_MARGIN = int((WINDOW_HEIGHT - (BOARD_HEIGHT * (BLOCK_SIZE + MARGIN_SIZE))) / 2 + TITLE_SIZE / 2)
TITLE_CENTER = (int(WINDOW_WIDTH / 2), int(Y_MARGIN / 2))

# Set Result
class Result(Enum):
    Win = 1
    Lost = 0

# Set Color
class Color(Enum):
    White = (255, 255, 255)
    DeepOrange = (234, 120, 33)
    Block0 = (204, 192, 179)
    Block2 = (238, 228, 218)
    Block4 = (237, 224, 200)
    Block8 = (242, 177, 121)
    Block16 = (244, 149, 99)
    Block32 = (245, 121, 77)
    Block64 = (245, 93, 55)
    Block128 = (238, 232, 99)
    Block256 = (237, 176, 77)
    Block512 = (236, 176, 77)
    Block1024 = (235, 148, 55)
    Block2048 = (234, 120, 33)
    TextLight = (255, 244, 234)
    TextDark = (119, 110, 101)


BACKGROUND_COLOR = Color.Block0.value
TEXT_COLOR = Color.TextDark.value
COLOR_SWITCHER = {
    0: Color.Block0.value,
    2: Color.Block2.value,
    4: Color.Block4.value,
    8: Color.Block8.value,
    16: Color.Block16.value,
    32: Color.Block32.value,
    64: Color.Block64.value,
    128: Color.Block128.value,
    256: Color.Block256.value,
    512: Color.Block512.value,
    1024: Color.Block1024.value,
    2048: Color.Block2048.value,
}


# Set Direction
class Direction(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3


# idx_boundary, idx_slide_start, idx_step, attr_name (col/line), attr_name (elem), range of another attr
SLIDE_SWITCHER = {
    Direction.Up: [0, BOARD_HEIGHT, 1, 'coordinate_x', 'coordinate_y', BOARD_WIDTH],
    Direction.Down: [BOARD_HEIGHT - 1, -1, -1, 'coordinate_x', 'coordinate_y', BOARD_WIDTH],
    Direction.Left: [0, BOARD_WIDTH, 1, 'coordinate_y', 'coordinate_x', BOARD_HEIGHT],
    Direction.Right: [BOARD_WIDTH - 1, -1, -1, 'coordinate_y', 'coordinate_x', BOARD_HEIGHT],
}


class Block:
    def __init__(self, x=random.randint(0, 3), y=random.randint(0, 3)):
        self.coordinate_x = x
        self.coordinate_y = y
        self.score = INIT_SCORE
        self.moved = False
        self.slide_enable = True
        self.next_coordinate_x = x
        self.next_coordinate_y = y


class Board:
    def __init__(self):
        self.blocks = [Block(), ]
        self.score = 0
        self.max_score = max([block.score for block in self.blocks])
        self.next_direction = Direction.Up
        self.is_end = False

    def handle_block_slide(self, direction):
        self.next_direction = direction
        # check each row/column (depend on direction)
        # TODO(Rundong) loop below can be optimised
        for line_col_idx in range(SLIDE_SWITCHER[direction][5]):
            # blocks in same row/column, idx in current row/column
            current_blocks = [(block, getattr(block, SLIDE_SWITCHER[direction][4])) for block in self.blocks if
                              getattr(block, SLIDE_SWITCHER[direction][3]) == line_col_idx]
            # search: [idx_boundary -> idx_move_start) by idx_step
            current_blocks.sort(key=lambda row: row[1], reverse=False if (SLIDE_SWITCHER[direction][2] == 1) else True)
            previous_idx = -1
            for block in current_blocks:
                if block[1] == SLIDE_SWITCHER[direction][0]:  # element on boundary
                    setattr(block[0], 'slide_enable', False)
                    setattr(block[0], 'next_' + SLIDE_SWITCHER[direction][3], line_col_idx)
                    setattr(block[0], 'next_' + SLIDE_SWITCHER[direction][4], SLIDE_SWITCHER[direction][0])
                else:  # not boundary element
                    if (block[1] - SLIDE_SWITCHER[direction][2] not in [row[1] for row in current_blocks]) or \
                            (getattr(current_blocks[previous_idx][0], 'slide_enable') is True):
                        setattr(block[0], 'slide_enable', True)
                        setattr(block[0], 'next_' + SLIDE_SWITCHER[direction][3], line_col_idx)
                        # calc next coordinate
                        if previous_idx == -1:
                            setattr(block[0], 'next_' + SLIDE_SWITCHER[direction][4], SLIDE_SWITCHER[direction][0])
                        else:
                            setattr(block[0], 'next_' + SLIDE_SWITCHER[direction][4],
                                    getattr(current_blocks[previous_idx][0],
                                            'next_' + SLIDE_SWITCHER[direction][4]) + SLIDE_SWITCHER[direction][2])
                assert ((0 <= block[0].next_coordinate_x < BOARD_WIDTH) and
                        (0 <= block[0].next_coordinate_y < BOARD_HEIGHT)), \
                    'invalid next move: ' + str((block[0].next_coordinate_x, block[0].next_coordinate_y)) + \
                    ' while current block is ' + str((block[0].coordinate_x, block[0].coordinate_y)) + '.'
                previous_idx += 1

    def slide_block(self):
        for block in self.blocks:
            if block.slide_enable:
                block.coordinate_x = block.next_coordinate_x
                block.coordinate_y = block.next_coordinate_y
                block.moved = True
                block.slide_enable = False

    def merge_block(self, direction):
        # TODO(Rundong) loop below can be optimised
        for line_col_idx in range(SLIDE_SWITCHER[direction][5]):
            current_blocks = [(block, getattr(block, SLIDE_SWITCHER[direction][4]))
                              for block in self.blocks if getattr(block, SLIDE_SWITCHER[direction][3]) == line_col_idx]
            if len(current_blocks) <= 1:
                continue
            else:
                # merge: (idx_move_start -> idx_boundary] by idx_step
                current_blocks.sort(key=lambda row: row[1],
                                    reverse=False if (SLIDE_SWITCHER[direction][2] == -1) else True)
                next_idx = 1
                for block in current_blocks:
                    if next_idx >= len(current_blocks):
                        break
                    else:
                        if getattr(block[0], 'score') == getattr(current_blocks[next_idx][0], 'score'):
                            setattr(current_blocks[next_idx][0], 'score', 2 * getattr(block[0], 'score'))
                            self.score += current_blocks[next_idx][0].score
                            self.blocks.remove(block[0])
                    next_idx += 1

    def generate_block(self):
        if len(self.blocks) >= (BOARD_WIDTH * BOARD_HEIGHT):
            return
        all_position = []
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                all_position.append((x, y))
        for block in self.blocks:
            (x, y) = (block.coordinate_x, block.coordinate_y)
            assert ((x, y) in all_position), 'contain block(s) with invalid location ' + str((x, y))
            all_position.remove((x, y))
        random.shuffle(all_position)
        position_insert = all_position.pop()
        self.blocks.append(Block(position_insert[0], position_insert[1]))

    def slide(self, direction):
        assert isinstance(direction, Direction), 'function "Block::slide": accepts "Direction" class argument only.'
        self.handle_block_slide(direction)
        self.slide_block()
        self.merge_block(direction)
        self.generate_block()
        
        if len(self.blocks) >= (BOARD_WIDTH * BOARD_HEIGHT):
            title = 'Pity, you LOST!\nPress "Esc" to QUIT\nor any other key to RESTART.'
            # handle_win_or_lost(Result.Lost, title)
            self.is_end = True
            
    def get_max_score(self):
        return max([block.score for block in self.blocks])

    def get_block_num(self):
        return len(self.blocks)


def get_state(main_board):
    # observation = self.get_observation(game)
    state = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
    for block in main_board.blocks:
        state[block.coordinate_y][block.coordinate_x] = math.log2(block.score)
    # print(state)
    return [state]


def main():
    global FPS_CLOCK, DISPLAY_SURF, FONT_OBJ, TITLE_OBJ, BLOCK_BOARD
    # pygame.init()
    # FPS_CLOCK = pygame.time.Clock()
    # DISPLAY_SURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    # BLOCK_BOARD = pygame.Rect(X_MARGIN - BOARD_OUTER_LINE_WIDTH, Y_MARGIN - BOARD_OUTER_LINE_WIDTH,
                              # (BLOCK_SIZE + MARGIN_SIZE) * BOARD_WIDTH + BOARD_OUTER_LINE_WIDTH * 2 - MARGIN_SIZE,
                              # (BLOCK_SIZE + MARGIN_SIZE) * BOARD_HEIGHT + BOARD_OUTER_LINE_WIDTH * 2 - MARGIN_SIZE)

    # Set Font
    # FONT_OBJ = pygame.font.Font('resource/SourceSansPro-Regular.ttf', FONT_SIZE)
    # TITLE_OBJ = pygame.font.Font('resource/SourceSansPro-Regular.ttf', TITLE_SIZE)

   
    title_text = 'Your Score: '

    # pygame.display.set_caption('2048')
    # DISPLAY_SURF.fill(BACKGROUND_COLOR)
    # FPS_CLOCK.tick(FPS)

    agent = Agent((4, 4), 4)
    action_map = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
    
    # Main game loop
    while True:
        agent.episodes = 0
        while agent.begin_episode():
            main_board = Board()
            # DISPLAY_SURF.fill(BACKGROUND_COLOR)
            # pygame.draw.rect(DISPLAY_SURF, Color.DeepOrange.value, BLOCK_BOARD, BOARD_OUTER_LINE_WIDTH)
            # pygame.event.get()
            while not main_board.is_end:
                score = main_board.score
                state = get_state(main_board)
                action_index = agent.get_action(state)
                action = action_map[action_index]
                main_board.slide(action)
                next_state = get_state(main_board)
                reward = main_board.score - score
                # memory = (state, action_index, reward, main_board.is_end, next_state)
                # print(memory)
                agent.commit_memory(state, action_index, reward, main_board.is_end, next_state)
                # draw_blocks(main_board)
                # pygame.display.update()
            agent.end_episode()

def draw_blocks(board_in):
    assert isinstance(board_in, Board), 'function "draw_blocks": accepts "Board" class argument only.'
    for block in board_in.blocks:
        left, top = block_position_to_pixel(block.coordinate_x, block.coordinate_y)
        block_rect_obj = pygame.Rect(left, top, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(DISPLAY_SURF, COLOR_SWITCHER[block.score], block_rect_obj)
        text_surface_obj = FONT_OBJ.render(str(block.score), True, TEXT_COLOR)
        text_rect_obj = text_surface_obj.get_rect()
        text_rect_obj.center = block_rect_obj.center
        DISPLAY_SURF.blit(text_surface_obj, text_rect_obj)


def block_position_to_pixel(x, y):
    assert (0 <= x < BOARD_WIDTH) and (0 <= y < BOARD_HEIGHT), \
        'function "block_position_to_pixel": invalid position ' + str((x, y)) + ', which must be in [0, board_size).'
    pixel_left = X_MARGIN + (BLOCK_SIZE + MARGIN_SIZE) * x
    pixel_top = Y_MARGIN + (BLOCK_SIZE + MARGIN_SIZE) * y

    return pixel_left, pixel_top


def draw_title(title):
    assert isinstance(title, str), 'function "draw_title": title must be string.'
    text_surface_obj = TITLE_OBJ.render(title, True, TEXT_COLOR)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.center = TITLE_CENTER
    DISPLAY_SURF.blit(text_surface_obj, text_rect_obj)
    

def handle_win_or_lost(result, title):
    assert isinstance(result, Result), 'function "handle_win_or_lost": result must be "Result" class.'
    DISPLAY_SURF.fill(BACKGROUND_COLOR)
    text_y = Y_MARGIN + int(TITLE_SIZE / 2)
    result_font = pygame.font.Font('resource/SourceSansPro-Regular.ttf', RESULT_SIZE)

    tokens = title.split('\n')
    text_surface_obj = TITLE_OBJ.render(tokens[0], True, TEXT_COLOR, BACKGROUND_COLOR)
    text_rect_obj = text_surface_obj.get_rect()
    text_rect_obj.top = text_y
    text_rect_obj.centerx = int(WINDOW_WIDTH / 2)
    DISPLAY_SURF.blit(text_surface_obj, text_rect_obj)
    text_y += RESULT_SIZE

    for text in tokens[1:]:
        text_y += int(RESULT_SIZE * 1.5)
        text_surface_obj = result_font.render(text, True, TEXT_COLOR, BACKGROUND_COLOR)
        text_rect_obj = text_surface_obj.get_rect()
        text_rect_obj.top = text_y
        text_rect_obj.centerx = int(WINDOW_WIDTH / 2)
        DISPLAY_SURF.blit(text_surface_obj, text_rect_obj)

if __name__ == '__main__':
    main()