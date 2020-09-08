"""Abstract class for neural networks."""

from abc import ABCMeta, abstractmethod, abstractproperty

from ..constants import DATA_TYPE_DEV
from ..eval.ResultList import ResultList


class BaseNeuralNetwork(object):
    """Abstract class for neural networks.

    All network classes need to inherit from this base class to achieve a unified
    interface for training and predicting.
    """
    __metaclass__ = ABCMeta

    @property
    @abstractproperty
    def config(self):
        """ExperimentConfig: The network configuration."""
        pass

    @abstractmethod
    def build(self):
        """Build the network with the chosen neural network library (Keras).

        Building a model from the configuration object provided in the constructor.
        The model may differ from network to network.

        Returns:
            keras.models.Model: The created model.
        """
        raise NotImplementedError("Must define `build` to use this base class.")

    @abstractmethod
    def train(self, epochs=None, verbose=True, evaluate_on_dev=True):
        """ Train the network with training data from the configuration.

        Args:
            epochs (int, optional): Number of epochs to train. This overwrites the settings from the configuration.
            verbose (bool, optional): Whether to print the progress. Defaults to True.
            evaluate_on_dev (bool, optional): Whether to evaluate the network on the development dataset after each
              epoch.

        Returns:
            None
        """
        raise NotImplementedError("Must define `train` to use this base class.")

    @abstractmethod
    def predict(self, sess, data_type=DATA_TYPE_DEV):
        """Perform prediction with the given model.

        Args:
            sess (object): Tensorflow session
            data_type (str): Which type of data to use (either dev or test)

        Returns:
            `list` of `tuple` of Task and ResultList: List of (task, result)-pairs that represent the prediction result
                for each task.
        """
        raise NotImplementedError("Must define `predict` to use this base class.")
