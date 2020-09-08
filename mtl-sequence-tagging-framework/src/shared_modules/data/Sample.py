"""Class to hold a single sample"""
import os

import numpy as np

from ..constants import TOKEN_PADDING


class Sample(object):

    def __init__(self, raw_tokens, raw_labels, tokens=None, labels=None, docid=None):
        """
        Initialize the sample object.
        A sample can hold only raw data or raw data and indexes. The `raw` property is an indicator.
        Args:
            raw_tokens (`list` of str): A list raw tokens (usually a sentence)
            raw_labels (`list` of str): A list of labels for the tokens
            tokens (`list` of int): A list of word indices
            labels (`list` of int): A list of label indices
            docid (string) : the document id for identifying the predictions
        """
        assert isinstance(raw_tokens, list)
        assert isinstance(raw_labels, list)
        assert (tokens is None and labels is None) or (tokens is not None and labels is not None)
        assert tokens is None or isinstance(tokens, list)
        assert labels is None or isinstance(labels, list)

        self._raw_tokens = raw_tokens
        self._raw_labels = raw_labels
        self._tokens = tokens
        self._labels = labels
        self._len = len(self._raw_tokens)
        self._raw = tokens is None and labels is None
        self._docid = docid

    def get_max_token_length(self):
        """
        Get the length of the longest token in the sequence of tokens for this sample.

        Returns:
            int: length of the longest token
        """
        return max([len(token) for token in self.raw_tokens])

    def get_tokens_as_char_ids(self, char2idx, token_length):
        """
        Generate a list of tokens represented by their character IDs from the raw tokens in this sample.
        Tokens are padded/cropped to the specified length

        Args:
            char2idx (`dict` of str): A mapping from chars to indices.
            token_length (int): The maximum length for a token

        Returns:
            np.ndarray: A matrix of  shape (sequence length, token length)
        """
        tokens = []

        for token in self.raw_tokens:
            # Crop token if it is too long
            if len(token) > token_length:
                token = token[:token_length]

            char_ids = [char2idx[char] for char in token]

            # Append padding if token is not long enough
            if len(char_ids) != token_length:
                char_ids += [char2idx[TOKEN_PADDING]] * (token_length - len(char_ids))

            tokens.append(char_ids)

        return np.asarray(tokens, dtype="int32")

    @property
    def raw_tokens(self):
        """

        Returns:
            `list` of str: A list raw tokens (usually a sentence)
        """
        return self._raw_tokens

    @property
    def raw_labels(self):
        """

        Returns:
            `list` of str: A list of labels for the tokens
        """
        return self._raw_labels

    @property
    def tokens(self):
        """

        Returns:
            `list` of int: A list of word indices
        """
        return self._tokens

    @property
    def tokens_as_array(self):
        """
        Converts the list of integers to a numpy array.
        Returns:
            np.ndarray: Token indices
        """
        return np.asarray(self.tokens, dtype="int32")

    @property
    def labels(self):
        """

        Returns:
            `list` of int: A list of label indices
        """
        return self._labels

    @property
    def labels_as_array(self):
        """
        Converts the list of integers to a numpy array.
        Returns:
            np.ndarray: Label indices
        """
        return np.asarray(self.labels, dtype="int32")

    @property
    def len(self):
        """

        Returns:
            int: length of the token list
        """
        return self._len

    @property
    def docid(self):
        """

        Returns:
            int: length of the token list
        """
        return self._docid

    @property
    def raw(self):
        """

        Returns:
            bool: whether or not the sample only contains raw data.
        """
        return self._raw

    def __str__(self):
        """
        Convert to CoNLL format of raw data.
        Returns:
            str: list of tokens and labels in CoNLL format
        """
        return os.linesep.join(["%s %s" % (token, label) for token, label in zip(self.raw_tokens, self.raw_labels)])
