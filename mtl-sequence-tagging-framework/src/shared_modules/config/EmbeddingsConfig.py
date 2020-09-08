"""Class for loading embeddings"""

import logging
from os import path
from gzip import open as gzopen

from .BaseConfig import BaseConfig


class EmbeddingsConfig(BaseConfig):
    def __init__(
            self,
            path,
            lower,
            separator,
            encoding,
            size,
            gzip
    ):
        """Initialize the embeddings configuration.

        Args:
            path (str): Path to embeddings file
            lower (bool): Whether or not to lowercase all words (usually False)
            separator (str): Separator used in the embeddings file (usually " ")
            encoding (str): Encoding of the file
            size (int): Number of dimensions of the embeddings
            gzip (bool): Whether or not the embeddings file is gzipped (usually True)
        """

        assert isinstance(path, str)
        assert isinstance(lower, bool)
        assert isinstance(separator, str)
        assert isinstance(encoding, str)
        assert isinstance(size, int)
        assert isinstance(gzip, bool)

        self._path = path
        self._lower = lower
        self._separator = separator
        self._encoding = encoding
        self._size = size
        self._gzip = gzip

        self._vectors = {}
        self._prepared = False
        self._paths_set = False
        self._paths = {}

    @property
    def path(self):
        """str: Path to embeddings file"""
        return self._path

    @property
    def lower(self):
        """bool: Whether or not to lowercase all words (usually False)"""
        return self._lower

    @property
    def separator(self):
        """str: Separator used in the embeddings file (usually " ")"""
        return self._separator

    @property
    def encoding(self):
        """str: Encoding of the file"""
        return self._encoding

    @property
    def size(self):
        """int: Number of dimensions of the embeddings"""
        return self._size

    @property
    def gzip(self):
        """bool: Whether or not the embeddings file is gzipped"""
        return self._gzip

    @property
    def vectors(self):
        """`dict` of `list` of float: A mapping from words to their corresponding word vectors"""
        return self._vectors

    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        logger = logging.getLogger("shared.embeddings_config.prepare")

        # Inspired by https://github.com/bplank/bilstm-aux/blob/master/src/lib/mio.py#L5-L22
        logger.debug("Opening embeddings file from %s with %s encoding", self.path, self.encoding)
        logger.debug("Using separator '%s'", self.separator)
        if self.lower:
            logger.debug("All words will be converted to lowercase")

        if self.gzip:
            f = gzopen(self.path, mode="r")
        else:
            f = open(self.path, mode="r")

        for line in f:
            try:
                fields = line.decode().strip().split(self.separator)
                # All fields but the first are values of the embedding vector
                vec = [float(value) for value in fields[1:]]
                # The first field is the word
                word = fields[0]

                # Apply lower case
                if self.lower:
                    word = word.lower()

                self._vectors[word] = vec

            except ValueError:
                logger.warn(
                    "Failed to prepare embeddings because line in embeddings file could not be read: %s",
                    line
                )
                return False

        # Close file
        f.close()

        # Check if the length of the vectors is actually the specified embeddings size
        logger.debug("Vectors should have dimensionality of %d", self.size)
        logger.debug("Vectors from embedding file have dimensionality of %d", len(vec) if vec else 0)

        assert len(vec) == self.size

        logger.info("Finished reading the embeddings file. Loaded vectors for %d distinct words.", len(self.vectors))

        self._prepared = True
        return True

    def to_dict(self):
        """
        Convert configuration to a dictionary.
        Returns:
            Dictionary with all the information of the instance.
        """
        logger = logging.getLogger("shared.embeddings_config.to_dict")

        logger.debug("Converting embeddings configuration for file %s to a dictionary.", self.path)
        return {
            "path": self.path,
            "lower": self.lower,
            "separator": self.separator,
            "encoding": self.encoding,
            "size": self.size,
            "gzip": self.gzip,
        }

    def sanity_check(self):
        """
        Check if the properties of the configuration object are valid.
        Returns:
            True in case of validity, False otherwise.
        """
        logger = logging.getLogger("shared.embeddings_config.sanity_check")

        path_valid = path.exists(self.path) and path.isfile(self.path)
        if not path_valid:
            logger.warn("Invalid path %s in embeddings configuration", self.path)

        return path_valid

    def set_paths(self, paths):
        """
        Set the paths for the experiment.
        Args:
            paths (`dict` of str): Necessary paths.
        """
        self._paths = paths
        self._paths_set = True

    @property
    def prepared(self):
        """
        Check if the configuration has been prepared. Should be true after `prepare` has been called.
        Returns:
            bool: True if the configuration has been prepared, False otherwise.
        """
        return self._prepared

    @property
    def paths_set(self):
        """
        Check if the paths have been set. Should be true after `set_paths` has been called.
        Returns:
            bool: True if the paths have been set, False otherwise.
        """
        return self._paths_set
