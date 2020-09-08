"""Factory to produce data readers based on the data format."""

from .BaseDataReader import BaseDataReader
from .ConllDataReader import ConllDataReader
from ..constants import CONLL


def create_data_reader(format=CONLL):
    """
    Create a data reader based the provided format.
    Raises an exception if no data reader exists for the format.

    Args:
        format (str): Data format

    Returns:
        BaseDataReader: Instance of the BaseDataReader
    """
    if format == CONLL:
        return ConllDataReader()
    else:
        raise Exception("Unknown data format '%s'" % format)
