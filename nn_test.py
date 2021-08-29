from nn_model import NeuralNetModel
from mcts import Node, SearchTree
import checkers.game

game = checkers.game.Game()
nn = NeuralNetModel()

print(game.moves)

root = Node(game, game.whose_turn())
tree = SearchTree(root, nn, [])
move = tree.get_move()

print(move.state.moves)
