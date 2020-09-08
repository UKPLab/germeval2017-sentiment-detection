"""Module for functions that build network layers"""

from keras.layers import ChainCRF, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.activations import softmax

from ..config.ExperimentConfig import ExperimentConfig
from ..config.TaskConfig import TaskConfig
from ..constants import CLASSIFIER_SOFTMAX


def terminate_task(shared_layer_output, task):
    """Terminate Task
    Terminate the provided task by sending the LSTM output through hidden layers
    first (if they are defined) and then sending the result to a softmax classifier.

    Args:
        shared_layer_output (object): Output of an LSTM layer.
        task (TaskConfig): Task configuration

    Returns:
        `tuple` of object: Reference to CRF layer, output layer, and task in case of CRF classifier, None, output
            layer, and task otherwise.
    """
    assert isinstance(task, TaskConfig)

    input_layer = shared_layer_output

    # Add hidden layers
    for i, hidden_layer_config in enumerate(task.hidden_layers):
        input_layer = Dense(
            units=hidden_layer_config.units,
            activation=hidden_layer_config.activation,
            name="hidden_%s_%d" % (task.name, i + 1)
        )(input_layer)

    if task.classifier == CLASSIFIER_SOFTMAX:
        # Add softmax layer
        return None, TimeDistributed(Dense(
            units=len(task.data_reader.get_labels()),
            activation=softmax
        ), name="softmax_output_%s" % task.name)(input_layer), task
    else:
        # Add dense layer to achieve the correct size
        input_layer = TimeDistributed(Dense(
            units=len(task.data_reader.get_labels())
        ))(input_layer)

        crf = ChainCRF(name="CRF_output_%s" % task.name)

        return crf, crf(input_layer), task


def build_embeddings_layer(config, trainable=False):
    """Build embeddings layer
    Build an embeddings layer from the provided configuration object.
    If trainable is set to True, the weights will be updated during training.

    Args:
        config (ExperimentConfig): Configuration object
        trainable (bool): Whether or not to update the weights during training. Defaults to False, i.e. no updates.

    Returns:
        A Keras embedding layer
    """
    assert isinstance(config, ExperimentConfig)

    if config.embeddings is None:
        # Default embeddings
        return Embedding(
            input_dim=config.vocab_size,
            output_dim=config.embedding_size,
            mask_zero=True,
            trainable=trainable
        )

    # Using pre-trained embeddings
    return Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_size,
        mask_zero=True,
        weights=[config.embedding_weights],
        trainable=trainable
    )
