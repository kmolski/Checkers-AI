# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pygame
from checkers.game import Game
from constants import WIDTH, HEIGHT, BLACK, RED, ROWS, COLS, SQUARE_SIZE, WHITE, PADDING, GREY, OUTLINE, BLUE, YELLOW


def draw_board(win):
    win.fill(BLACK)
    for row in range(ROWS):
        for col in range(row % 2, COLS, 2):
            pygame.draw.rect(win, WHITE, (row * SQUARE_SIZE, col * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def abs_pos_piece(piece):
    abs_pos = piece.position - 1
    row = abs_pos // 4
    col = abs_pos % 4 * 2 + row % 2
    return SQUARE_SIZE * col + SQUARE_SIZE // 2, SQUARE_SIZE * row + SQUARE_SIZE // 2


def draw_piece(win, piece):
    if not piece.captured:
        radius = SQUARE_SIZE // 2 - PADDING
        x, y = abs_pos_piece(piece)
        color = RED if piece.player == 1 else BLUE
        pygame.draw.circle(win, GREY, (x, y), radius + OUTLINE)
        pygame.draw.circle(win, color, (x, y), radius)
    if piece.king:
        pygame.draw.circle(win, YELLOW, (x,y), radius // 2)


def draw(win):
    draw_board(win)
    for piece in game.board.pieces:
        draw_piece(WIN, piece)

    pygame.display.update()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    game = Game()
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Checkers')
    run = True
    while run:
        draw(WIN)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                game.move(game.get_possible_moves()[0]) #lol algorytm bardzo naiwny

        if game.is_over():
            run = False
            print('Game over.')
            print(f'Player {game.get_winner()} won')


    pygame.quit()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
