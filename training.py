import gc
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

DEFAULT_TOURNAMENT_COUNT = 200
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

            del inputs
            del win_values
            del action_ps
            gc.collect()

        logging.info(f"Training session {self.session_index}: saving weights")
        nn_model.persist_weights_to_file()

        del training_data
        del nn_model
        gc.collect()

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
            del game_training_data

        return training_data

    def play_until_complete(self, game, nn_model):
        prev_boards = []
        training_data = []

        agent1 = NeuralNetAgent(game, nn_model=nn_model)
        agent2 = NeuralNetAgent(game, nn_model=nn_model)

        while not game.is_over():
            current_player = agent1 if game.whose_turn() == 1 else agent2
            prev_boards.append(game.board)

            move, node = current_player.get_next_move(prev_boards)

            training_data.append(
                {
                    "input": encode_game_state(game, prev_boards)[0],
                    "player": current_player,
                    "action_ps": encode_action_ps(node),
                }
            )

            game.move(move)

            agent1.use_new_state(node)
            agent2.use_new_state(node)

        logging.info(
            f"Training session {self.session_index}: game.moves_since_last_capture: "
            + f"{game.moves_since_last_capture}, game.get_winner(): {game.get_winner()}"
        )

        del prev_boards
        return training_data, game.get_winner()

    @classmethod
    def train(cls, session_count=DEFAULT_TRAINING_SESSIONS):
        for i in range(session_count):
            BaseTrainingSession(i).run()


class TournamentSession:
    def __init__(self, session_index, game_count):
        super().__init__()
        self.game_count = game_count

        self.session_index = session_index
        self.c_name = ""
        self.game_index = 0
        self.c_turn = None

    def run(self, best):
        logging.info(f"Tournament {self.session_index}: gathering training data")
        training_data, _ = self.play_games(best, best)
        shuffle(training_data)

        training_data_len = len(training_data)
        chunk_size = get_chunk_size(training_data_len)
        chunk_count = training_data_len // chunk_size

        logging.info(
            f"Tournament {self.session_index}: training challenger neural net "
            + f"with {chunk_count} chunks of data, {chunk_size} elements each"
        )

        challenger_nn = NeuralNetModel(keras_model=best.model)
        for (index, chunk) in enumerate(chunks(training_data, chunk_size)):
            inputs = [datum["input"] for datum in chunk]
            win_values = [datum["win_value"] for datum in chunk]
            action_ps = [datum["action_ps"] for datum in chunk]

            logging.info(f"Tournament {self.session_index}: processing chunk {index}")
            challenger_nn.train(inputs, win_values, action_ps)

            del inputs
            del win_values
            del action_ps
            gc.collect()

        del training_data
        gc.collect()

        logging.info(f"Tournament {self.session_index}: challenging the best net")
        _, win_count = self.play_games(best, challenger_nn)

        return challenger_nn, win_count

    def play_games(self, best, challenger):
        training_data = []
        challenger_win_count = 0

        for i in range(self.game_count):
            logging.info(f"Tournament {self.session_index}: playing game {i}")
            self.game_index = i

            game = Game()
            game_training_data, winner = self.play_until_complete(
                game, best, challenger
            )

            if winner == self.c_turn:
                challenger_win_count += 1

            for datum in game_training_data:
                datum["win_value"] = adjust_score(datum["player"], winner)
            training_data.extend(game_training_data)
            del game_training_data

        return training_data, challenger_win_count

    def play_until_complete(self, game, best, challenger):
        prev_boards = []
        training_data = []

        self.c_turn = self.game_index % 2 + 1
        challenger_agent = NeuralNetAgent(game, nn_model=challenger)
        best_agent = NeuralNetAgent(game, nn_model=best)

        while not game.is_over():
            current = challenger_agent if game.whose_turn() == self.c_turn else best_agent
            prev_boards.append(game.board)

            move, node = current.get_next_move(prev_boards)

            training_data.append(
                {
                    "input": encode_game_state(game, prev_boards)[0],
                    "player": current,
                    "action_ps": encode_action_ps(node),
                }
            )

            game.move(move)

            challenger_agent.use_new_state(node)
            best_agent.use_new_state(node)

        logging.info(
            f"Tournament {self.session_index}: " +
            f"moves since last capture: {game.moves_since_last_capture}, " +
            f"winner: {'challenger' if game.get_winner() == self.c_turn else 'best'}"
        )

        del prev_boards
        return training_data, game.get_winner()

    def get_best_net(self):
        logging.info(f"Tournament: picking the best net")

        best, *challengers = [
            NeuralNetModel(p) for p in Path(".").glob("data/base_training/*")
        ]

        challengers_with_scores = []
        for c in challengers:
            self.c_name = c.weights_file
            logging.info(f"Tournament: playing games against {self.c_name}")

            _, challenger_win_count = self.play_games(best, c)
            challengers_with_scores.append((c, challenger_win_count))

        logging.info(f"Tournament: picking best net")
        challengers_with_scores.sort(key=lambda it: it[1])
        largest_win_count = challengers_with_scores[-1][1]

        if largest_win_count / self.game_count > 0.5:
            best = challengers_with_scores.pop()[0]

        logging.info(f"Tournament: got best net: {best.weights_file}")
        return best

    @classmethod
    def train(
        cls,
        tournament_count=DEFAULT_TOURNAMENT_COUNT,
        game_count=DEFAULT_GAMES_IN_TOURNAMENT,
    ):
        best = TournamentSession(None, game_count).get_best_net()

        best.weights_file = join("data", "tournaments", "best_init_weights")
        best.persist_weights_to_file()

        for i in range(tournament_count):
            (challenger, wins) = TournamentSession(i, game_count).run(best)

            logging.info(f"Tournament {i}: challenger win rate: {wins / game_count}")
            if wins / game_count > 0.55:
                logging.info(f"Tournament {i}: got new best net")
                del best
                best = challenger

            gc.collect()

        best.weights_file = join("data", "tournaments", "best_weights")
        best.persist_weights_to_file()
