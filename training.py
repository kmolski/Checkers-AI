import logging
from itertools import islice
from os.path import join
from pathlib import Path
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
DEFAULT_GAMES_IN_TOURNAMENT = 20


def chunks(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk


def get_chunk_size(data_len):
    return min(data_len // DEFAULT_CHUNK_COUNT, DEFAULT_MAX_CHUNK_SIZE)


class BaseTrainingSession:
    def __init__(self, session_index, game_count=DEFAULT_TRAINING_GAME_COUNT):
        super().__init__()
        self.game_count = game_count
        self.weights_file = join("data", "base_training", f"weights_{session_index}")

        self.session_index = session_index
        self.game_index = 0

    def run(self):
        logging.info(f"Training session {self.session_index}: starting")

        nn_model = NeuralNetModel()
        nn_model.weights_file = self.weights_file

        training_data = self.play_games(nn_model)
        shuffle(training_data)

        training_data_len = len(training_data)
        chunk_size = get_chunk_size(training_data_len)
        chunk_count = training_data_len // chunk_size

        logging.info(
            f"Training session {self.session_index}: training neural net "
            + f"with {chunk_count} chunks of data, {chunk_size} elements each"
        )
        for (index, chunk) in enumerate(chunks(training_data, chunk_size)):
            inputs = [datum["input"] for datum in chunk]
            win_values = [datum["win_value"] for datum in chunk]
            action_ps = [datum["action_ps"] for datum in chunk]

            logging.info(
                f"Training session {self.session_index}: processing chunk {index}"
            )
            nn_model.train(inputs, win_values, action_ps)

        logging.info(f"Training session {self.session_index}: saving weights")
        nn_model.persist_weights_to_file()

    def play_games(self, nn_model):
        training_data = []
        for i in range(self.game_count):
            logging.info(f"Training session {self.session_index}: playing game {i}")
            self.game_index = i

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
            current_player = agent1 if game.whose_turn() == 1 else agent2
            move, node = current_player.get_next_move(prev_boards)

            training_data.append(
                {
                    "input": encode_game_state(game, prev_boards)[0],
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
        for i in range(session_count):
            BaseTrainingSession(i).run()


class TournamentSession:
    def __init__(self, session_index, game_count=DEFAULT_GAMES_IN_TOURNAMENT):
        super().__init__()
        self.game_count = game_count

        self.session_index = session_index
        self.c_name = ""
        self.game_index = 0

    def run(self, best, challengers):
        logging.info(f"Tournament {self.session_index}: starting")

        challengers_with_scores = []
        for c in challengers:
            self.c_name = c.weights_file
            logging.info(f"Tournament {self.session_index}: playing games against {self.c_name}")

            training_data, challenger_win_count = self.play_games(best, c)
            shuffle(training_data)

            training_data_len = len(training_data)
            chunk_size = get_chunk_size(training_data_len)
            chunk_count = training_data_len // chunk_size

            logging.info(
                f"Tournament {self.session_index}, challenger {self.c_name}: training neural net "
                + f"with {chunk_count} chunks of data, {chunk_size} elements each"
            )
            for (index, chunk) in enumerate(chunks(training_data, chunk_size)):
                inputs = [datum["input"] for datum in chunk]
                win_values = [datum["win_value"] for datum in chunk]
                action_ps = [datum["action_ps"] for datum in chunk]

                logging.info(
                    f"Tournament {self.session_index}, challenger {self.c_name}: processing chunk {index}"
                )
                best.train(inputs, win_values, action_ps)
                c.train(inputs, win_values, action_ps)

            challengers_with_scores.append((c, challenger_win_count))
        return challengers_with_scores

    def play_games(self, best, challenger):
        training_data = []
        challenger_win_count = 0

        for i in range(self.game_count):
            logging.info(f"Tournament {self.session_index}, challenger {self.c_name}: playing game {i}")
            self.game_index = i

            game = Game()
            game_training_data, winner = self.play_until_complete(game, best, challenger)

            if winner == 1:
                challenger_win_count += 1

            for datum in game_training_data:
                datum["win_value"] = adjust_score(datum["player"], winner)
            training_data.extend(game_training_data)

        return training_data, challenger_win_count

    def play_until_complete(self, game, best, challenger):
        prev_boards = []
        training_data = []

        challenger_agent = NeuralNetAgent(game, nn_model=challenger)
        best_agent = NeuralNetAgent(game, nn_model=best)

        while not game.is_over():
            current_player = challenger_agent if game.whose_turn() == 1 else best_agent
            move, node = current_player.get_next_move(prev_boards)

            training_data.append(
                {
                    "input": encode_game_state(game, prev_boards)[0],
                    "player": current_player,
                    "action_ps": encode_action_ps(node),
                }
            )

            prev_boards.append(game.board)
            game.move(move)

            challenger_agent.use_new_state(node)
            best_agent.use_new_state(node)

        return training_data, game.get_winner()

    @classmethod
    def train(cls, tournament_count=DEFAULT_TOURNAMENT_COUNT):
        best, *challengers = [NeuralNetModel(p) for p in Path(".").glob("data/base_training/*")]

        for i in range(tournament_count):
            logging.info(f"Tournament {i}: starting")
            challengers_with_scores = TournamentSession(i).run(best, challengers)

            logging.info(f"Tournament {i}: picking new best net")
            challengers_with_scores.sort(key=lambda it: it[1])
            largest_win_count = challengers_with_scores[-1][1]
            if largest_win_count / DEFAULT_GAMES_IN_TOURNAMENT > 0.6:
                new_best = challengers_with_scores.pop()[0]
                logging.info(f"Tournament {i}: got new best net: {new_best.weights_file}")
                challengers = [c[0] for c in challengers_with_scores] + [best]
                best = new_best

        best.weights_file = join("data", "tournaments", "best_weights")
        best.persist_weights_to_file()
