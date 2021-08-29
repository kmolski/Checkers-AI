import time

from checkers.game import Game

from agents import ComputerAgent
from mcts import adjust_score
from nn_model import NeuralNetModel

DEFAULT_MAX_BATCH_SIZE = 1024
DEFAULT_BATCH_COUNT = 25

DEFAULT_TRAINING_SESSIONS = 10
DEFAULT_TRAINING_GAME_COUNT = 100

DEFAULT_TOURNAMENT_COUNT = 10
DEFAULT_GAMES_IN_TOURNAMENT = 100


class TrainingSession:
    def __init__(self, session_index, game_count=DEFAULT_TRAINING_GAME_COUNT):
        self.game_count = game_count
        self.nn_model = NeuralNetModel(
            weights_file=f"data/training/weights_{time.strftime('%Y%m%d_%H%M%S')}_{session_index}"
        )

    def train_neural_net(self):
        training_data = self.play_games()


    def play_games(self):
        training_data = []
        for i in range(self.game_count):
            game = Game()
            result = self.play_until_complete(game)
            training_data.extend(result)
        return training_data

    def play_until_complete(self, game):
        game_training_data = []
        player1 = ComputerAgent(game)
        player2 = ComputerAgent(game)

        while not game.is_over():
            pass

        return game_training_data

    @classmethod
    def train(cls):
        for i in range(DEFAULT_TRAINING_SESSIONS):
            pass


class TournamentSession:
    def __init__(self, game_count=DEFAULT_GAMES_IN_TOURNAMENT):
        self.game_count = game_count

    @classmethod
    def train(cls):
        pass
