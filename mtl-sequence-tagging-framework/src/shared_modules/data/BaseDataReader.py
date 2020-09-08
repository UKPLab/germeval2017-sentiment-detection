"""Abstract class for data readers."""

from abc import ABCMeta, abstractproperty, abstractmethod

from numpy import ndarray

from ..config.FileConfig import FileConfig


class BaseDataReader(object):
    """Abstract class for data readers.

    All data readers have to inherit from this class to allow for a unified data access.
    Data formatting and preprocessing is performed in the subclasses.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def add_files(self, files):
        """Add files to this data reader.

        Args:
            files (`dict` of FileConfig): A dictionary of files that maps file configs to train, dev, and test.

        Returns:
            None
        """
        raise NotImplementedError("Must define `add_files` to use this base class.")

    @abstractmethod
    def add_paths(self, paths):
        """
        Add the paths for the experiment.
        Args:
            paths (`dict` of str): Necessary paths.
        """
        raise NotImplementedError("Must define `add_paths` to use this base class")

    @abstractmethod
    def add_name(self, name):
        """
        Set a name for this data reader. This is necessary for unique pickle files.
        Args:
            name (str): A unique name (e.g. the task name)
        """
        raise NotImplementedError("Must define `add_name` to use this base class")

    @property
    @abstractproperty
    def files(self):
        """
            `dict` of FileConfig: List of file configs for the data.
            This is a list even if only one file name was supplied in `add_files`.
        """
        pass

    @abstractmethod
    def get_data(self, data_type="train", out_format="padded", word2idx=None):
        """Get the data from the corpus.

        Get the data from the corpus. The user can choose the type and format of the data.
        The data format choice allows to get data that is ready for consumption by a Keras
        network.

        Args:
            data_type (str): The type of data. One of "train", "dev" or "test".
            out_format (str): The format of the data. One of "padded" (words represented by
                their indices and padded by the index for <MASK>), "padded-right" (as "padded",
                but padding is applied from the right and not from the left) "index" (words represented
                by their indices) or "raw" (words in their original form).
            word2idx (`dict` of int): Mapping from words to indices. Only used for "padded"
                and "index". If not supplied, using `self.get_words(out_format="word2idx")`.

        Returns:
            `tuple` of ndarray or `tuple` of `list` of int or `tuple` of `list` of str: The data.
        """
        raise NotImplementedError("Must define `get_data` to use this base class.")

    @abstractmethod
    def get_words(self, out_format="list"):
        """Get all words.

        Depending on `format`, either get a list of all words (over all files) or a
        dictionary of all words in the corpus mapped to indices.
        The list/dict of words should contain the <MASK> word (at index 0).

        Args:
            out_format (str): The output format. One of "list" or "word2idx".

        Returns:
            `list` of str: `dict` of int: A word list or dictionary.
        """
        raise NotImplementedError("Must define `get_words` to use this base class.")

    @abstractmethod
    def get_labels(self, out_format="list"):
        """Get all labels.

        Depending on `format`, either get a list of all labels (over all files) or a
        dictionary of all labels in the corpus mapped to indices.
        The list/dict of words should contain the <MASK> label (at index 0).

        Args:
            out_format (str): The output format. One of "list" or "label2idx".

        Returns:
            `list` of str: `dict` of int: A label list or dictionary.
        """
        raise NotImplementedError("Must define `get_labels` to use this base class.")

    @abstractmethod
    def get_max_sequence_length(self):
        """Determine the maximum sequence length.

        This method determines the maximum sequence length over all documents in the corpus.
        This could be, for instance, the length of the longest sentence in the corpus.

        Returns:
            int: Maximum sequence length.
        """
        raise NotImplementedError("Must define `get_max_sequence_length` to use this base class.")
