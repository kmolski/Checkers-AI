import logging

from nn_model import NeuralNetModel
from mcts import Node, SearchTree
import checkers.game

### NEURAL NET

game = checkers.game.Game()
nn = NeuralNetModel()

print(game.moves)

root = Node(game, game.whose_turn())
tree = SearchTree(root, nn)
move = tree.get_next_move([])

print(move.state.moves)

### BASE TRAINING

from training import BaseTrainingSession

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

BaseTrainingSession.train()
