"""Class to store a single batch"""
import numpy as np


class Batch(object):
    def __init__(self, labels, tokens, samples, characters=None):
        """
        Initialize the batch object.

        Args:
            labels (np.ndarray): A matrix of labels with shape (batch size, sequence length)
            tokens (np.ndarray): A matrix of tokens with shape (batch size, sequence length)
            samples (`list` of Sample): A list of samples that serve as a source for the label and token matrices.
            characters (np.ndarray, optional): A matrix of characters with shape
                (batch size, sequence length, word length)
        """
        self._characters = characters
        self._samples = samples
        self._tokens = tokens
        self._labels = labels

        # Calculated values
        # shape = (batch size)
        self._sequence_lengths = np.ones(labels.shape[0]) * labels.shape[1]
        if characters is not None:
            # shape = (batch size, sequence length)
            self._word_lengths = np.ones((characters.shape[0], characters.shape[1])) * characters.shape[2]
        else:
            self._word_lengths = None

    @property
    def characters(self):
        return self._characters

    @property
    def samples(self):
        return self._samples

    @property
    def tokens(self):
        return self._tokens

    @property
    def labels(self):
        return self._labels

    @property
    def sequence_lengths(self):
        return self._sequence_lengths

    @property
    def word_lengths(self):
        return self._word_lengths
