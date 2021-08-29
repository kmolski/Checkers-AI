import logging
from itertools import islice
from multiprocessing import Process
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

DEFAULT_TOURNAMENT_COUNT = 20
DEFAULT_GAMES_IN_TOURNAMENT = 100


def chunks(it, size):
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk


def get_chunk_size(data_len):
    return min(data_len // DEFAULT_CHUNK_COUNT, DEFAULT_MAX_CHUNK_SIZE)


class BaseTrainingSession(Process):
    def __init__(self, session_index, game_count=DEFAULT_TRAINING_GAME_COUNT):
        super().__init__()
        self.game_count = game_count
        self.session_index = session_index
        self.weights_file = join("data", "base_training", f"weights_{session_index}")

    def run(self):
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
        )

        logging.info(f"Starting training session {self.session_index}")
        logging.info("Gathering game data...")

        nn_model = NeuralNetModel()
        nn_model.weights_file = self.weights_file
        training_data = self.play_games(nn_model)

        training_data_len = len(training_data)
        shuffle(training_data)

        logging.info("Training the neural net...")
        for chunk in chunks(training_data, get_chunk_size(training_data_len)):
            inputs = (datum["input"] for datum in chunk)
            win_values = (datum["win_value"] for datum in chunk)
            action_ps = (datum["action_ps"] for datum in chunk)

            logging.info("Processing training data chunk...")
            nn_model.train(inputs, win_values, action_ps)

        logging.info("Saving neural net weights...")
        nn_model.persist_weights_to_file()

    def play_games(self, nn_model):
        training_data = []
        for i in range(self.game_count):
            logging.info(f"Playing game no. {i}")
            game = Game()
            game_training_data, winner = self.play_until_complete(game, nn_model)

            for datum in game_training_data:
                datum["win_value"] = adjust_score(datum["player"], winner)
            training_data.extend(game_training_data)

        return training_data

    def play_until_complete(self, game, nn_model):
        prev_boards = []
        training_data = []

        agent1 = NeuralNetAgent(game, nn_model=nn_model)
        agent2 = NeuralNetAgent(game, nn_model=nn_model)

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
        processes = [BaseTrainingSession(i) for i in range(session_count)]

        [p.start() for p in processes]
        [p.join() for p in processes]


class TournamentSession:
    def __init__(self, game_count=DEFAULT_GAMES_IN_TOURNAMENT):
        self.game_count = game_count

    @classmethod
    def train(cls):
        pass
