from checkers import Game

class TrainingSession:
    def __init__(self, game_count):
        self.game_count = game_count

    def start(self):
        games = []
        for i in range(self.game_count):
            # TODO: Rough draft, develop this further
            game = Game()
            player1 = None
            player2 = None
            self.play_game(game, player1, player2)
            games.append(game)

    def play_game(self, game, player1, player2):
        pass