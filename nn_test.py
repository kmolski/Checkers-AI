import logging

from nn_model import NeuralNetModel
from mcts import Node, SearchTree
import checkers.game

from training import BaseTrainingSession
from training import TournamentSession

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
)

# NEURAL NET TEST

# game = checkers.game.Game()
# nn = NeuralNetModel()
#
# print(game.moves)
#
# tree = SearchTree(Node(game, game.whose_turn()), nn)
# move = tree.get_next_move([])
#
# print(move.state.moves)

# BASE TRAINING

# BaseTrainingSession.train()

# TOURNAMENTS

TournamentSession.train()
