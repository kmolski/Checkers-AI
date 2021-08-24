from nn_model import NeuralNetModel
from mcts import Node, SearchTree
from checkers.game import Game

game = Game()
nn = NeuralNetModel()

root = Node(game, game.whose_turn())
tree = SearchTree(root, nn, [])
move = tree.get_move()

print(move.state.moves)