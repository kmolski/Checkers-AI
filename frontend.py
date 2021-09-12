import random
import time

import pygame
from checkers.game import Game
from constants import WIDTH, HEIGHT, BLACK, RED, ROWS, COLS, SQUARE_SIZE, WHITE, PADDING, GREY, OUTLINE, BLUE, YELLOW, \
    GREEN

from agents import NeuralNetAgent


class Frontend:
    def __init__(self, real_players):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Checkers')

        self.game = Game()
        self.running = True
        self.selected_piece = None
        self.possible_move_targets = []
        self.real_players = real_players

        self.nn_agent = NeuralNetAgent(self.game, weights_file="data/base_training/weights_5")
        self.prev_boards = []

    def loop(self):
        while self.running:
            self._update_caption()
            self._draw()

            if self.game.is_over():
                self.running = False
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.button)

        if self.game.is_over():
            pygame.display.update()
            time.sleep(3)

    def _draw(self):
        self._draw_board()
        for piece in self.game.board.pieces:
            self._draw_piece(piece)
        for target in self.possible_move_targets:
            self._draw_target(target)

        pygame.display.update()

    def _draw_board(self):
        self.screen.fill(BLACK)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(self.screen, WHITE, (row * SQUARE_SIZE, col * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def _draw_piece(self, piece):
        if not piece.captured:
            radius = SQUARE_SIZE // 2 - PADDING
            x, y = self._abs_pos(piece.position)
            color = RED if piece.player == 1 else BLUE
            if piece == self.selected_piece:
                pygame.draw.circle(self.screen, GREEN, (x, y), radius + 10)

            pygame.draw.circle(self.screen, GREY, (x, y), radius + OUTLINE)
            pygame.draw.circle(self.screen, color, (x, y), radius)
            if piece.king:
                pygame.draw.circle(self.screen, YELLOW, (x, y), radius // 2)

    def _draw_target(self, target):
        radius = SQUARE_SIZE // 2
        x, y = self._abs_pos(target)
        pygame.draw.circle(self.screen, GREEN, (x, y), radius // 3)

    def _abs_pos(self, pos):
        abs_pos = pos - 1
        row = abs_pos // 4
        col = abs_pos % 4 * 2 + (row + 1) % 2
        return SQUARE_SIZE * col + SQUARE_SIZE // 2, SQUARE_SIZE * row + SQUARE_SIZE // 2

    def _mouse_pos_to_square(self, pos):
        x, y = pos
        row = y // SQUARE_SIZE
        col = x // SQUARE_SIZE
        if row % 2 == 0:
            if col % 2 == 0:
                return None
            else:
                col = col // 2
        else:
            if col % 2 == 0:
                col = col // 2
            else:
                return None

        return row * 4 + col + 1

    def _find_piece(self, square):
        for piece in self.game.board.pieces:
            if piece.position == square:
                return piece

        return None

    def _handle_click(self, button):
        self.prev_boards.append(self.game.board)

        if self.game.whose_turn() not in self.real_players:
            self._ai_move()
            return

        #rightclick
        if button == 2:
            self.game.move(random.choice(self.game.get_possible_moves()))
            return

        pos = pygame.mouse.get_pos()
        square = self._mouse_pos_to_square(pos)
        piece = self._find_piece(square)
        if piece is not None:
            self._select(piece)
            return

        if square in self.possible_move_targets:
            move = [self.selected_piece.position, square]
            node = self.nn_agent.get_node_for_move(move)
            self.game.move(move)
            self.possible_move_targets = []

            self.nn_agent.use_new_state(node)

    def _update_caption(self):
        if self.game.is_over():
            pygame.display.set_caption(f'Checkers - {self._player_to_str(self.game.get_winner())} won!')
            print(f'Checkers - player {self._player_to_str(self.game.get_winner())} won!')
        else:
            pygame.display.set_caption(f'Checkers - {self._player_to_str(self.game.whose_turn())}')

    def _player_to_str(self, player_num):
        if player_num == 1:
            return "RED"
        elif player_num == 2:
            return "BLUE"
        else:
            return "NOBODY"

    def _select(self, piece):
        if piece.player == self.game.whose_turn():
            self.selected_piece = piece
            moves = piece.get_possible_positional_moves() + piece.get_possible_capture_moves()
            legal_moves = [m for m in self.game.get_possible_moves() if m in moves]
            self.possible_move_targets = [m[1] for m in legal_moves]

    def _ai_move(self):
        move, node = self.nn_agent.get_next_move(self.prev_boards)
        self.game.move(move)
        self.nn_agent.use_new_state(node)
