import frontend


if __name__ == '__main__':
    real_players = [1]
    frontend = frontend.Frontend(real_players)
    frontend.loop()
