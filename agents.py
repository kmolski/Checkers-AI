from mcts import Node, SearchTree
from nn_model import NeuralNetModel


class NeuralNetAgent:
    def __init__(self, game, nn_model=None, weights_file=None):
        self.game = game
        self.nn_model = nn_model or NeuralNetModel(weights_file)

        root_node = Node(game, game.whose_turn())
        self.mcts = SearchTree(root_node, self.nn_model)

    def get_next_move(self, prev_boards):
        moves = self.game.get_possible_moves()

        if len(moves) == 1:
            return moves[0], self.get_node_for_move(moves[0])
        else:
            node = self.mcts.get_next_move(prev_boards)
            return node.state.moves[-1], node

    def use_new_state(self, node):
        self.mcts.root_node = node

    def get_node_for_move(self, move):
        try:
            root_children = self.mcts.root_node.children
            return next(filter(lambda n: n.state.moves[-1] == move, root_children))
        except StopIteration:
            return Node(self.mcts.root_node.state, None).move(move)


class HumanAgent:
    def __init__(self, game):
        self.game = game
