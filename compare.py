import random

from checkers.game import Game

from agents import NeuralNetAgent
from mcts import Node
from nn_model import NeuralNetModel

NN_WINS = 0
NN_MODEL = NeuralNetModel("data/tournaments/best_weights")

for i in range(100):
    game = Game()
    nn_agent = NeuralNetAgent(game, nn_model=NN_MODEL)

    prev_boards = []
    while not game.is_over():
        current_player = "RANDOM" if game.whose_turn() == 1 else nn_agent
        prev_boards.append(game.board)
        if current_player == "RANDOM":
            move = random.choice(game.get_possible_moves())
            node = Node(game, None).move(move)
            game.move(move)
            nn_agent.use_new_state(node)
        else:
            move, node = nn_agent.get_next_move(prev_boards)
            game.move(move)
            nn_agent.use_new_state(node)
    if game.get_winner() != 1:
        NN_WINS += 1
    print(f"{'random' if game.get_winner() == 1 else 'neural net'} won")

print(f"neural net win rate: {NN_WINS}%")
