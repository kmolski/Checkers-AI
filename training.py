from itertools import islice
from random import shuffle

from checkers.game import Game

from agents import ComputerAgent
from mcts import adjust_score
from nn_model import NeuralNetModel

DEFAULT_MAX_CHUNK_SIZE = 1152
DEFAULT_CHUNK_COUNT = 24

DEFAULT_TRAINING_SESSIONS = 10
DEFAULT_TRAINING_GAME_COUNT = 100

DEFAULT_TOURNAMENT_COUNT = 10
DEFAULT_GAMES_IN_TOURNAMENT = 100


def chunks(it, size):
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk


def get_chunk_size(data_len):
    return min(data_len // DEFAULT_CHUNK_COUNT, DEFAULT_MAX_CHUNK_SIZE)


class BaseTrainingSession:
    def __init__(self, session_index, game_count=DEFAULT_TRAINING_GAME_COUNT):
        self.game_count = game_count
        self.weights_file = f"data/training/weights_{session_index}"

    def train_neural_net(self):
        training_data = self.play_games()

        training_data_len = len(training_data)
        shuffle(training_data)

        nn_model = NeuralNetModel()
        nn_model.weights_file = self.weights_file

        for chunk in chunks(training_data, get_chunk_size(training_data_len)):
            inputs = (components[0] for components in chunk)
            win_values = (components[1] for components in chunk)
            action_ps = (components[2] for components in chunk)

            nn_model.train(inputs, win_values, action_ps)

        nn_model.persist_weights_to_file()

    def play_games(self):
        training_data = []
        for i in range(self.game_count):
            game = Game()
            result = self.play_until_complete(game)

            # TODO: adjust the score based on the winner

            training_data.extend(result)
        return training_data

    def play_until_complete(self, game):
        training_data = []
        agent1 = ComputerAgent(game)
        agent2 = ComputerAgent(game)

        while not game.is_over():
            current_player = agent1 if game.whose_turn() == 1 else agent2
            current_player.get_next_move()
            pass

        return training_data

    @classmethod
    def train(cls):
        # TODO: parallelize this with multiprocessing
        for i in range(DEFAULT_TRAINING_SESSIONS):
            BaseTrainingSession(i).train_neural_net()


class TournamentSession:
    def __init__(self, game_count=DEFAULT_GAMES_IN_TOURNAMENT):
        self.game_count = game_count

    @classmethod
    def train(cls):
        pass
