"""Class for the configuration of a hidden layer"""

import logging

from .BaseConfig import BaseConfig


class HiddenLayerConfig(BaseConfig):
    """Class for configuring a hidden layer

    Most configuration options are equivalent to the Dense layer of
    Keras (https://keras.io/layers/core/).
    """

    def __init__(
            self,
            units,
            activation,
            use_bias,
            dropout_keep_probability
    ):
        """Initialize the hidden layer configuration.

        Args:
            units (int): Number of hidden units
            activation (str or callable): The name of an activation function or a function itself
            use_bias (bool): Whether or not tu use a bias vector
            dropout_keep_probability (float): Keep probability for dropout_keep_probability. Between 0.0 and 1.0
        """

        assert isinstance(units, int)
        assert isinstance(activation, str) or callable(activation)
        assert isinstance(use_bias, bool)
        assert isinstance(dropout_keep_probability, float) and 0.0 <= dropout_keep_probability <= 1.0

        self._units = units
        self._activation = activation
        self._use_bias = use_bias
        self._dropout_keep_probability = dropout_keep_probability

        self._prepared = False
        self._paths = {}
        self._paths_set = False

    @property
    def units(self):
        """int: Number of hidden units"""
        return self._units

    @property
    def activation(self):
        """str: callable: The name of an activation function or a function itself"""
        return self._activation

    @property
    def use_bias(self):
        """bool: Whether or not tu use a bias vector"""
        return self._use_bias

    @property
    def dropout_keep_probability(self):
        """float: Keep probability for dropout_keep_probability. Between 0.0 and 1.0"""
        return self._dropout_keep_probability

    def prepare(self):
        self._prepared = True
        return True

    def sanity_check(self):
        logger = logging.getLogger("shared.hidden_layer_config.sanity_check")

        enough_units = self._units > 0
        if not enough_units:
            logger.warn("Not enough units. Specified: %d", self._units)

        valid_activation = callable(self._activation) or self._activation in [
            "softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"
        ]

        if not valid_activation:
            logger.warn("Invalid activation function supplied %s", self._activation)

        return enough_units and valid_activation

    def to_dict(self):
        return {
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout_keep_probability": self.dropout_keep_probability,
        }

    @property
    def prepared(self):
        """
        Check if the configuration has been prepared. Should be true after `prepare` has been called.
        Returns:
            bool: True if the configuration has been prepared, False otherwise.
        """
        return self._prepared

    def set_paths(self, paths):
        """
        Set the paths for the experiment.
        Args:
            paths (`dict` of str): Necessary paths.
        """
        self._paths = paths
        self._paths_set = True

    @property
    def paths_set(self):
        """
        Check if the paths have been set. Should be true after `set_paths` has been called.
        Returns:
            bool: True if the paths have been set, False otherwise.
        """
        return self._paths_set
