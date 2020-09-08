"""Data preprocessing methods"""
import re

import numpy as np

from ..config import EmbeddingsConfig
from ..constants import TOKEN_DATE, TOKEN_NUMBER, TOKEN_TIME


def word_normalize(word):
    """
    Normalize the provided word.
    Copied from NR's code (`wordNormalize` in WordEmbeddings.py).
    Args:
        word (str): Word to normalize

    Returns:
        Normalized word
    """
    word = word.lower()
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)
    word = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}", TOKEN_DATE, word)
    word = re.sub("[0-9]{2}:[0-9]{2}:[0-9]{2}", TOKEN_TIME, word)
    word = re.sub("[0-9]{2}:[0-9]{2}", TOKEN_TIME, word)
    word = re.sub("[0-9.,]+", TOKEN_NUMBER, word)
    return word


def merge_embeddings(embedding_configurations):
    """
    Merge multiple embeddings with each other.
    Args:
        embedding_configurations (`list` of EmbeddingsConfig): a list of embedding configurations.

    Returns:
        `dict` of np.ndarray: A mapping from words to word vectors
    """

    assert all([isinstance(config, EmbeddingsConfig.EmbeddingsConfig) for config in embedding_configurations])

    if len(embedding_configurations) == 1:
        return embedding_configurations[0].vectors

    all_words = set()

    for config in embedding_configurations:
        all_words.update(set(config.vectors.keys()))

    vectors = {}

    # Concatenate vectors for all words that are contained in all embeddings
    for word in all_words:
        if all([word in config.vectors for config in embedding_configurations]):
            vectors[word] = np.concatenate(tuple(config.vectors[word] for config in embedding_configurations))

    return vectors
