"""Class for wrapping a file"""

from os import path
import logging

from .BaseConfig import BaseConfig


class FileConfig(BaseConfig):
    def __init__(self, path, scheme="IOB", word_column=0, label_column=1, column_separator=" ", encoding="utf8"):
        """Initialize the file configuration.

        Args:
            path (str): Path to the file
            scheme (str): Which tagging scheme the file is using, e.g. IOB or IOBES
            word_column (int): In which column the word can be found
            label_column (int): In which column the label can be found (useful for multi-label files)
            column_separator (str): Which character or which characters separate the columns from each other
            encoding (str): Encoding of the file
        """
        # Ensure that data types are correct
        assert isinstance(path, str)
        assert isinstance(scheme, str)
        assert isinstance(word_column, int)
        assert isinstance(label_column, int)
        assert column_separator == "space" or column_separator == "tab"
        assert isinstance(encoding, str)

        self._column_separator = " " if column_separator == "space" else "\t"
        self._word_column = word_column
        self._label_column = label_column
        self._scheme = scheme
        self._path = path
        self._encoding = encoding

        self._prepared = False
        self._paths = {}
        self._paths_set = False

    @property
    def column_separator(self):
        """str: Which character or which characters separate the columns from each other"""
        return self._column_separator

    @property
    def word_column(self):
        """int: In which column the word can be found"""
        return self._word_column

    @property
    def label_column(self):
        """int: In which column the label can be found (useful for multi-label files)"""
        return self._label_column

    @property
    def scheme(self):
        """str: Which tagging scheme the file is using, e.g. IOB or IOBES"""
        return self._scheme

    @property
    def path(self):
        """str: Path to the file"""
        return self._path

    @property
    def filename(self):
        return path.basename(self.path)

    @property
    def encoding(self):
        """str: Encoding of the file"""
        return self._encoding

    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        self._prepared = True
        return True

    def sanity_check(self):
        logger = logging.getLogger("shared.file_config.sanity_check")
        path_valid = path.exists(self._path) and path.isfile(self._path)
        if not path_valid:
            logger.warn("Invalid path %s in file configuration", self._path)

        return path_valid

    def to_dict(self):
        return {
            "path": self.path,
            "scheme": self.scheme,
            "word_column": self.word_column,
            "label_column": self.label_column,
            "column_separator": self.column_separator,
            "encoding": self.encoding,
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
