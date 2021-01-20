import os
from tensorflow.keras import layers
from tensorflow.keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def only_simple_rnn():
    model = models.Sequential([
        layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        layers.SimpleRNN(20, return_sequences=True),
        layers.SimpleRNN(1)])

    return model


def simple_rnn(dense_node):
    model = models.Sequential([
        layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        layers.SimpleRNN(20),
        layers.Dense(dense_node)])

    return model


def simple_rnn_2(dense_node):
    model = models.Sequential([
        layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        layers.SimpleRNN(20, return_sequences=True),
        # 모든 timestep에 Dense(10) 적용
        layers.TimeDistributed(layers.Dense(dense_node))
    ])

    return model
