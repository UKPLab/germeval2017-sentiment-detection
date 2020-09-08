"""Constant module

This module contains all common constants.
"""

# Token types
TOKEN_PADDING = "<__PADDING__>"
TOKEN_UNKNOWN = "<__UNKNOWN__>"
TOKEN_DATE = "<__DATE__>"
TOKEN_TIME = "<__TIME__>"
TOKEN_NUMBER = "<__NUMBER__>"

DOCSTART = "DOCSTART"

# Data formats
CONLL = "CONLL"

# Data output formats
DATA_OUT_RAW = "raw"
DATA_OUT_INDEX = "index"
DATA_OUT_PADDED = "padded"
DATA_OUT_PADDED_RIGHT = "padded-right"

# Data types
DATA_TYPE_TRAIN = "train"
DATA_TYPE_DEV = "dev"
DATA_TYPE_TEST = "test"

# Classifiers
CLASSIFIER_SOFTMAX = "softmax"
CLASSIFIER_CRF = "CRF"

# Directories
DIR_OUT = "out"
DIR_PKL = "pkl"
DIR_SRC = "src"
DIR_DATA = "data"
DIR_RUN = "run-%03d"
DIR_MODEL_WEIGHTS = "saved_model"
DIR_PREDICTION_OUT = "predictions"
DIR_TENSOR_BOARD = "tensor_board"
DIR_BATCHES_OUT = "batches"

PREFIX_MODEL_WEIGHTS = "model.weights"

# List length alignment strategies
ALIGNMENT_STRATEGY_RANDOM_SAMPLE = "random"
ALIGNMENT_STRATEGY_CROP = "crop"

# RNN
RNN_UNIT_TYPE_SIMPLE = "simple"
RNN_UNIT_TYPE_GRU = "GRU"
RNN_UNIT_TYPE_LSTM = "LSTM"

# Activation functions
ACTIVATION_TANH = "tanh"
ACTIVATION_LINEAR = "linear"
ACTIVATION_RELU = "relu"
ACTIVATION_SIGMOID = "sigmoid"

# Optimizers
OPTIMIZER_SGD = "SGD"
OPTIMIZER_ADAM = "adam"
OPTIMIZER_ADAGRAD = "adagrad"
OPTIMIZER_ADADELTA = "adadelta"
VALID_OPTIMIZERS = [
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAGRAD,
    OPTIMIZER_ADADELTA,
]

# Encoding types
ENCODING_NONE = "NONE"
ENCODING_BIO = "BIO"
ENCODING_IOB = "IOB"
ENCODING_IOBES = "IOBES"

# Metrics
METRIC_ACCURACY = "accuracy"
METRIC_F1 = "f1"
METRIC_F1_O = "f1_o"
METRIC_F1_B = "f1_b"
METRIC_PRECISION = "precision"
METRIC_PRECISION_O = "precision_o"
METRIC_PRECISION_B = "precision_b"
METRIC_RECALL = "recall"
METRIC_RECALL_O = "recall_o"
METRIC_RECALL_B = "recall_b"
METRIC_AM_COMPONENTS_05 = "am_components_0.5"
METRIC_AM_COMPONENTS_0999 = "am_components_0.999"
METRIC_AM_RELATIONS_05 = "am_relations_0.5"
METRIC_AM_RELATIONS_0999 = "am_relations_0.999"
METRIC_WORD_ACCURACY = "word_accuracy"
METRIC_AVG_EDIT_DISTANCE = "avg_edit_distance"
METRIC_MEDIAN_EDIT_DISTANCE = "median_edit_distance"

VALID_METRICS = [
    METRIC_ACCURACY,
    METRIC_F1,
    METRIC_F1_O,
    METRIC_F1_B,
    METRIC_PRECISION,
    METRIC_PRECISION_O,
    METRIC_PRECISION_B,
    METRIC_RECALL,
    METRIC_RECALL_O,
    METRIC_RECALL_B,
    METRIC_AM_COMPONENTS_05,
    METRIC_AM_COMPONENTS_0999,
    METRIC_AM_RELATIONS_05,
    METRIC_AM_RELATIONS_0999,
    METRIC_WORD_ACCURACY,
    METRIC_AVG_EDIT_DISTANCE,
    METRIC_MEDIAN_EDIT_DISTANCE,
]

# Character level information network type
CHAR_LSTM = "LSTM"
CHAR_CNN = "CNN"

# Task type
TASK_TYPE_GENERIC = "GENERIC"
TASK_TYPE_AM = "AM"
