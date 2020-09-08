"""Mappings between constants and tensorflow object"""

import tensorflow as tf

from shared_modules.constants import ACTIVATION_LINEAR, ACTIVATION_RELU, ACTIVATION_SIGMOID, ACTIVATION_TANH, \
    OPTIMIZER_SGD, OPTIMIZER_ADAM, OPTIMIZER_ADAGRAD, OPTIMIZER_ADADELTA, RNN_UNIT_TYPE_LSTM, RNN_UNIT_TYPE_GRU, \
    RNN_UNIT_TYPE_SIMPLE

ACTIVATION_MAPPING = {
    ACTIVATION_RELU: tf.nn.relu,
    ACTIVATION_LINEAR: None,
    ACTIVATION_SIGMOID: tf.nn.sigmoid,
    ACTIVATION_TANH: tf.nn.tanh,
}

OPTIMIZER_MAPPING = {
    OPTIMIZER_SGD: tf.compat.v1.train.GradientDescentOptimizer,
    OPTIMIZER_ADAM: tf.compat.v1.train.AdamOptimizer,
    OPTIMIZER_ADAGRAD: tf.compat.v1.train.AdagradOptimizer,
    OPTIMIZER_ADADELTA: tf.compat.v1.train.AdadeltaOptimizer,
}

RNN_CELL_MAPPING = {
    RNN_UNIT_TYPE_SIMPLE: tf.compat.v1.nn.rnn_cell.BasicRNNCell,
    RNN_UNIT_TYPE_GRU: tf.compat.v1.nn.rnn_cell.GRUCell,
    RNN_UNIT_TYPE_LSTM: tf.compat.v1.nn.rnn_cell.LSTMCell,
}
