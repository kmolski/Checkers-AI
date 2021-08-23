import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, regularizers
from tensorflow.keras.layers import (
    add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
)
from tensorflow.keras.models import load_model, Model as KerasModel
from tensorflow.keras.optimizers import SGD

# 32 pieces, 8 last game states, 4 different piece types (white/black X men/kings)
# 1 (8x4) layer to describe the current player
# 1 (8x4) layer to encode the number of moves without a capture
INPUT_DIMENSIONS = (34, 8, 4)
OUTPUT_DIMENSIONS = 8 * 8 * 4
RESIDUAL_LAYER_COUNT = 11
CONV_KERNEL_COUNT = 75
CONV_KERNEL_SIZE = (4, 4)
LEARNING_RATE = 0.1
MOMENTUM = 0.9

DEFAULT_REGULARIZER = regularizers.l2(0.0001)
CONV2D_DEFAULT_OPTIONS = {
    "data_format": "channels_first",
    "padding": "same",
    "use_bias": False,
    "activation": "linear",
    "kernel_regularizer": DEFAULT_REGULARIZER,
}


def create_batch_norm_and_leaky_relu(layer):
    layer = BatchNormalization(axis=1)(layer)
    return LeakyReLU()(layer)


def create_hidden_conv(layer):
    return Conv2D(
        filters=CONV_KERNEL_COUNT,
        kernel_size=CONV_KERNEL_SIZE,
        **CONV2D_DEFAULT_OPTIONS
    )(layer)


def create_convolutional(layer):
    layer = create_hidden_conv(layer)
    return create_batch_norm_and_leaky_relu(layer)


def create_residual(layer):
    init_layer = layer
    layer = create_hidden_conv(layer)
    layer = create_batch_norm_and_leaky_relu(layer)

    layer = create_hidden_conv(layer)
    layer = BatchNormalization(axis=1)(layer)

    layer = add([init_layer, layer])
    return LeakyReLU()(layer)


def create_value_head(layer):
    layer = Conv2D(filters=1, kernel_size=(1, 1), **CONV2D_DEFAULT_OPTIONS)(layer)

    layer = create_batch_norm_and_leaky_relu(layer)
    layer = Flatten()(layer)
    layer = Dense(
        20, use_bias=False, activation="linear", kernel_regularizer=DEFAULT_REGULARIZER
    )(layer)

    layer = LeakyReLU()(layer)
    return Dense(
        1,
        use_bias=False,
        activation="tanh",
        kernel_regularizer=DEFAULT_REGULARIZER,
        name="value_head",
    )(layer)


def create_policy_head(layer):
    layer = Conv2D(filters=2, kernel_size=(1, 1), **CONV2D_DEFAULT_OPTIONS)(layer)

    layer = create_batch_norm_and_leaky_relu(layer)
    layer = Flatten()(layer)
    return Dense(
        OUTPUT_DIMENSIONS,
        use_bias=False,
        activation="linear",
        kernel_regularizer=DEFAULT_REGULARIZER,
        name="policy_head",
    )(layer)


def softmax_cross_entropy_with_logits(y_true, y_pred):
    predictions = y_pred
    labels = y_true

    zero = tf.zeros(shape=tf.shape(labels), dtype=tf.float32)
    where = tf.equal(labels, zero)

    negatives = tf.fill(tf.shape(labels), -100.0)
    predictions = tf.where(where, negatives, predictions)

    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions)


class NeuralNetModel:
    def __init__(self, weights_file=None):
        backend.clear_session()

        if weights_file is not None:
            self.weights_file = weights_file
            self.model = load_model(
                weights_file,
                custom_objects={
                    "softmax_cross_entropy_with_logits": softmax_cross_entropy_with_logits
                },
            )
        else:
            self.weights_file = f"weights_{time.strftime('%Y%m%d_%H%M%S')}"
            input_layer = Input(shape=INPUT_DIMENSIONS)

            # Convolutional layer
            shared_layers = create_convolutional(input_layer)

            # Shared residual layers
            for _ in range(RESIDUAL_LAYER_COUNT):
                shared_layers = create_residual(shared_layers)

            # Value head
            value_head = create_value_head(shared_layers)

            # Policy head
            policy_head = create_policy_head(shared_layers)

            self.model = KerasModel(
                inputs=[input_layer], outputs=[value_head, policy_head]
            )
            self.model.compile(
                optimizer=SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM),
                loss={
                    "value_head": "mean_squared_error",
                    "policy_head": softmax_cross_entropy_with_logits,
                },
                loss_weights={"value_head": 0.5, "policy_head": 0.5},
            )

    def persist_weights_to_file(self):
        self.model.save(self.weights_file)

    def clear_keras_session(self):
        backend.clear_session()

    def train(self, inputs, win_values, action_ps):
        output = {
            "value_head": np.array(win_values),
            "policy_head": np.array(action_ps),
        }
        self.model.fit(np.array(inputs), output)

    def predict(self, input_data):
        output = self.model.predict(input_data)
        return {"win_value": output[0][0][0], "action_ps": output[1].tolist()[0]}
