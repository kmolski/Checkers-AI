from encoder import align_coordinates


def get_coordinates_from_position(position, width):
    return (position - 1) % width, (position - 1) // width


def get_move_value(game, move, action_ps):
    turn = game.whose_turn()
    board = game.board

    from_coords = get_coordinates_from_position(move[0], board.width)
    from_x, from_y = align_coordinates(*from_coords, turn, board)

    to_coords = get_coordinates_from_position(move[1], board.width)
    to_x, to_y = align_coordinates(*to_coords, turn, board)

    diff_x, diff_y = (to_x - from_x, to_y - from_y)
    board_tiles = board.width * board.height

    is_capture = abs(diff_y) == 2
    towards_south = diff_y > 0
    if is_capture:
        towards_east = diff_x == 1
    else:
        even_to_odd_y = (from_y % 2 == 0)
        towards_east = (diff_x == 1) if even_to_odd_y else (diff_x == 0)

    index = (
        from_x
        + (from_y * board.width)
        + board_tiles * (towards_east * 1 + towards_south * 2 + is_capture * 4)
    )
    return action_ps[index]
