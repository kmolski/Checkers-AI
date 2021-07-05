from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add

# 32 pieces, 7 last game states, 4 different piece types (white/black X men/kings)
# 1 (7x4) layer to describe the current player
# 1 (7x4) layer to encode the number of moves without a capture
DEFAULT_NET_DIMENSIONS = (34, 7, 4)

class NeuralNetModel:
    def __init__(self):
        pass
    def persist_weights_to_file(self):
        pass
    def clear_keras_session(self):
        pass
    def train(self):
        pass
    def predict(self):
        pass