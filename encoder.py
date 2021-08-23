from itertools import product

import numpy as np

from nn_model import INPUT_DIMENSIONS


def align_coordinates(x, y, turn, board):
    return (x, y) if turn != 1 else (board.width - x - 1, board.height - y - 1)


def get_layer_for_piece(piece, turn):
    return ((piece.player != turn) * 2 + piece.king * 1) * 8


def encode_board_state(input_data, offset, turn, board_state):
    for (y, x) in product(range(board_state.height), range(board_state.width)):
        pos = board_state.position_layout[y][x]
        piece = board_state.searcher.get_piece_by_position(pos)
        if piece:
            layer_index = get_layer_for_piece(piece, turn)
            (x, y) = align_coordinates(x, y, turn, board_state)
            input_data[layer_index + offset][y][x] = 1


def encode_latest_board_states(turn, latest_boards):
    input_data = np.zeros(INPUT_DIMENSIONS, dtype=np.int)
    for index, board in enumerate(reversed(latest_boards)):
        encode_board_state(input_data, index, turn, board)
    return input_data


def encode_turns_without_capturing_moves(game):
    moves = game.moves_since_last_capture
    digit_list_len = game.board.height * game.board.width
    bin_digit_list = [int(d) for d in list(f"{moves:b}".zfill(digit_list_len))]
    return np.reshape(bin_digit_list, INPUT_DIMENSIONS[1:])


def encode_player_turn(turn):
    if turn != 1:
        return np.ones(INPUT_DIMENSIONS[1:], dtype=np.int)
    return np.zeros(INPUT_DIMENSIONS[1:], dtype=np.int)


def encode_game_state(game, board_states):
    # Encode the 8 last board states
    input_data = encode_latest_board_states(game.whose_turn(), board_states[-8:])
    # TODO: try to switch the following and see what happens:
    # Encode player turn
    input_data[32] = encode_player_turn(game.whose_turn())
    # Encode the number of turns without capturing moves
    input_data[33] = encode_turns_without_capturing_moves(game)

    return input_data
