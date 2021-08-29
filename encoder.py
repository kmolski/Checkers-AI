from itertools import product

import numpy as np

from nn_model import INPUT_DIMENSIONS


def align_coordinates(x, y, turn, board):
    return (x, y) if turn != 1 else (board.width - x - 1, board.height - y - 1)


def get_channel_for_piece(piece, turn):
    return ((piece.player != turn) * 2 + piece.king * 1) * 8


def encode_board_state(
    input_data, player_turn, turns_with_no_capture, offset, turn, board_state
):
    for (y, x) in product(range(board_state.height), range(board_state.width)):
        pos = board_state.position_layout[y][x]
        piece = board_state.searcher.get_piece_by_position(pos)
        if piece:
            layer_index = get_channel_for_piece(piece, turn)
            (x, y) = align_coordinates(x, y, turn, board_state)
            input_data[y][x][layer_index + offset] = 1
        input_data[y][x][32] = player_turn[y][x]
        input_data[y][x][33] = turns_with_no_capture[y][x]


def encode_turns_without_capturing_moves(game):
    moves = game.moves_since_last_capture
    digit_list_len = game.board.height * game.board.width
    bin_digit_list = [int(d) for d in list(f"{moves:b}".zfill(digit_list_len))]
    return np.reshape(bin_digit_list, INPUT_DIMENSIONS[:-1])


def encode_player_turn(turn):
    if turn != 1:
        return np.ones(INPUT_DIMENSIONS[:-1], dtype=np.int)
    return np.zeros(INPUT_DIMENSIONS[:-1], dtype=np.int)


def encode_game_state(game, board_states):
    turn = game.whose_turn()
    input_data = np.zeros(INPUT_DIMENSIONS, dtype=np.int)

    player_turn = encode_player_turn(turn)
    turns_with_no_capture = encode_turns_without_capturing_moves(game)

    # Encode the 8 last board states
    for index, board in enumerate(reversed(board_states[-8:])):
        encode_board_state(
            input_data, player_turn, turns_with_no_capture, index, turn, board
        )

    return np.array([input_data])
