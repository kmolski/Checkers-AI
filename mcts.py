from enum import Enum, auto
from functools import partial
from math import log, sqrt
from random import choice


def ucs1_evaluator(exploration_factor, node):
    return (node.win_count / node.visit_count +
            exploration_factor * sqrt(log(node.parent.visit_count) / node.visit_count))


def make_selector(evaluator):
    return lambda nodes: max(nodes, key=evaluator)


DEFAULT_EXPLORATION = sqrt(2)
DEFAULT_SELECTOR = make_selector(partial(ucs1_evaluator, DEFAULT_EXPLORATION))


class NodeType(Enum):
    Leaf = auto()


class Node:
    def __init__(self, state, player):
        self.state = state
        self.player = player

        self.win_count = 0
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

    def select_most_visited_child(self, node_selector):
        return max(self.children, key=lambda node: node.visit_count)

    def propagate_value(self, value):
        self.win

class Tree:
    def __init__(self, root_node, node_selector):
        self.root_node = root_node
        self.node_selector = node_selector

    def select_best_move(self):
        return self.root_node.select_most_visited_child()

    def select_(self):
        visit_values = map(lambda child: child.visit_count, self.root_node.children)
        pro