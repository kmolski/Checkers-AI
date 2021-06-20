from functools import partial
from math import log, sqrt
from random import choice


def max_elements(iterable, key=lambda x: x):
    it = iter(iterable)
    max_item = next(it)
    max_item_key = key(max_item)
    result = [max_item]

    for item in it:
        item_key = key(item)
        if item_key > max_item_key:
            max_item_key = item_key
            result = [item]
        elif item_key == max_item_key:
            result.append(item)

    return result


def ucs1_evaluator(exploration_factor, root_player, node):
    # https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
    move_strength = node.net_score / max(1, node.visit_count)
    exploration_part = (exploration_factor * (node.policy_value or 1) *
                        sqrt(log(node.parent.visit_count) / max(1, node.visit_count)))
    return (1 if root_player == node.parent.player else -1) * move_strength + exploration_part


DEFAULT_EXPLORATION = sqrt(2)
DEFAULT_EVALUATOR = partial(ucs1_evaluator, DEFAULT_EXPLORATION)


class Node:
    def __init__(self, state, player):
        self.state = state
        self.player = player

        self.net_score = 0
        self.visit_count = 0
        self.policy_value = None

        self.parent = None
        self.children = []

    def is_leaf(self):
        return len(self.children) < 1

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def select_best_child(self, root_player, node_evaluator=DEFAULT_EVALUATOR):
        node_evaluator = partial(node_evaluator, root_player)
        preferred = max_elements(self.children, node_evaluator)
        return choice(preferred)

    def select_most_visited_child(self):
        most_visited = max_elements(self.children, key=lambda node: node.visit_count)
        return choice(most_visited)

    def add_and_propagate_score(self, score):
        self.net_score += score
        self.visit_count += 1

        if self.parent is not None:
            self.parent.add_and_propagate_score(score)

    def expand(self, root_player, nn_model, prev_boards):
        if self.state.is_over():
            winner = self.state.get_winner()
            score = 1 if winner == root_player else 0 if winner is None else -1
            self.add_and_propagate_score(score)
        else:
            # TODO: use the NN model to predict the score and policy values
            # TODO: update the net score of the current node with the prediction
            # TODO: add legal moves to the node and get policy values for them
            pass


DEFAULT_ROLLOUTS_PER_MOVE = 50


class Tree:
    def __init__(self, root_node, nn_model, prev_moves, rollouts_per_move=DEFAULT_ROLLOUTS_PER_MOVE):
        self.root_node = root_node
        self.nn_model = nn_model
        self.rollouts_per_move = rollouts_per_move

    def get_leaf(self):
        node = self.root_node
        while not node.is_leaf():
            node = node.select_best_child(self.root_node.player)
        return node

    def do_rollouts(self, input_builder):
        for i in range(self.rollouts_per_move):
            self.get_leaf().expand(self.nn_model, input_builder)

    def get_move(self):
        self.do_rollouts()
        return self.root_node.select_most_visited_child()
