import logging
from itertools import islice
from os.path import join
from random import shuffle

from checkers.game import Game

from agents import NeuralNetAgent
from encoder import encode_game_state, encode_action_ps
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
        self.weights_file = join("data", "base_training", f"weights_{session_index}")

    def train_neural_net(self):
        logging.info("Gathering game data...")
        training_data = self.play_games()

        training_data_len = len(training_data)
        shuffle(training_data)

        nn_model = NeuralNetModel()
        nn_model.weights_file = self.weights_file

        logging.info("Training the neural net...")
        for chunk in chunks(training_data, get_chunk_size(training_data_len)):
            inputs = (datum["input"] for datum in chunk)
            win_values = (datum["win_value"] for datum in chunk)
            action_ps = (datum["action_ps"] for datum in chunk)

            logging.info("Processing training data chunk...")
            nn_model.train(inputs, win_values, action_ps)

        logging.info("Saving neural net weights...")
        nn_model.persist_weights_to_file()

    def play_games(self):
        training_data = []
        for i in range(self.game_count):
            logging.info(f"Playing game no. {i}")
            game = Game()
            game_training_data, winner = self.play_until_complete(game)

            for datum in game_training_data:
                datum["win_value"] = adjust_score(datum["player"], winner)
            training_data.extend(game_training_data)

        return training_data

    def play_until_complete(self, game):
        prev_boards = []
        training_data = []

        agent1 = NeuralNetAgent(game)
        agent2 = NeuralNetAgent(game)

        while not game.is_over():
            logging.info(f"Making a move: {game.get_possible_moves()}")
            current_player = agent1 if game.whose_turn() == 1 else agent2
            move, node = current_player.get_next_move(prev_boards)

            training_data.append(
                {
                    "input": encode_game_state(game, prev_boards),
                    "player": current_player,
                    "action_ps": encode_action_ps(node),
                }
            )

            prev_boards.append(game.board)
            game.move(move)

            agent1.use_new_state(node)
            agent2.use_new_state(node)

        return training_data, game.get_winner()

    @classmethod
    def train(cls, session_count=DEFAULT_TRAINING_SESSIONS):
        # TODO: parallelize this with multiprocessing
        for i in range(session_count):
            logging.info(f"Starting training session {i}")
            BaseTrainingSession(i).train_neural_net()


class TournamentSession:
    def __init__(self, game_count=DEFAULT_GAMES_IN_TOURNAMENT):
        self.game_count = game_count

    @classmethod
    def train(cls):
        pass
