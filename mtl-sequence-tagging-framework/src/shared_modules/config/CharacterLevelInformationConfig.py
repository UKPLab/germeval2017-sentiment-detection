"""Class for configuring character level infomration"""

from .BaseConfig import BaseConfig
from ..constants import CHAR_CNN, CHAR_LSTM


class CharacterLevelInformationConfig(BaseConfig):

    def __init__(self, network_type, dimensionality, hidden_units):
        """Initialize the character level information configuration.

        Args:
            network_type (str): Which type of network to use for character level information (LSTM or CNN).
            dimensionality (int): Dimensionality of character embeddings
            hidden_units (int): Number of hidden units (only necessary for LSTM extractor)
        """
        print (network_type)
        assert network_type in [CHAR_CNN, CHAR_LSTM]
        assert isinstance(dimensionality, int) and dimensionality > 0
        assert isinstance(hidden_units, int) and hidden_units > 0

        self._network_type = network_type
        self._dimensionality = dimensionality
        self._hidden_units = hidden_units

        self._prepared = False
        self._paths = {}
        self._paths_set = False

    @property
    def network_type(self):
        """str: Which type of network to use for character level information (LSTM or CNN)."""
        return self._network_type

    @property
    def dimensionality(self):
        """int: Dimensionality of character embeddings"""
        return self._dimensionality

    @property
    def hidden_units(self):
        """int: Number of hidden units (only necessary for LSTM extractor)"""
        return self._hidden_units

    def to_dict(self):
        return {
            "network_type": self.network_type,
            "dimensionality": self.dimensionality,
            "hidden_units": self.hidden_units
        }

    def sanity_check(self):
        return True

    def set_paths(self, paths):
        """
        Set the paths for the experiment.
        Args:
            paths (`dict` of str): Necessary paths.
        """
        self._paths = paths
        self._paths_set = True

    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        self._prepared = True
        return True

    @property
    def paths_set(self):
        """
        Check if the paths have been set. Should be true after `set_paths` has been called.
        Returns:
            bool: True if the paths have been set, False otherwise.
        """
        return self._paths_set

    @property
    def prepared(self):
        """
        Check if the configuration has been prepared. Should be true after `prepare` has been called.
        Returns:
            bool: True if the configuration has been prepared, False otherwise.
        """
        return self._prepared
