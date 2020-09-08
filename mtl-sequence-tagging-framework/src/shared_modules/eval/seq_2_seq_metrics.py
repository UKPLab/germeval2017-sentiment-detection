"""
Metrics for sequence-to-sequence (seq2seq) tasks, in particular grapheme-to-phoneme (G2P).
"""

import editdistance
from numpy import median

TOKEN_EMPTY = "EMPTY"
TOKEN_MYJOIN = "_MYJOIN_"


def character_list_to_string(character_list):
    """
    Convert a list of characters to a string by concatenating them.
    Moreover, the placeholders EMPTY and _MYJOIN_ will be removed.

    Args:
        character_list (`list` of str or `list` of unicode): A list of characters. Characters are not necessarily of
            length 1 because of joined characters, e.g. s_MYJOIN_c

    Returns:
        str or unicode: Cleaned string
    """
    assert isinstance(character_list, list)
    assert all([isinstance(char, str) or isinstance(char, unicode) for char in character_list])

    word = "".join(character_list)
    # Remove EMPTY and _MYJOIN_
    word = word.replace(TOKEN_EMPTY, "")
    word = word.replace(TOKEN_MYJOIN, "")

    return word


def word_accuracy(predicted_words, true_words):
    """
    Calculate the word accuracy which is the fraction of words that have been "translated" (e.g. mapped to their
    phoneme) correctly (100% match between prediction and truth).

    The tokens EMPTY and _MYJOIN_ are ignored.

    Args:
        predicted_words (`list` of `list` of str or `list` of `list` of unicode): A list of predicted words represented
            as lists of characters.
        true_words (`list` of `list` of str or `list` of `list` of unicode): A list of true words represented as lists
            of characters.

    Returns:
        float: the word accuracy, i.e. the fraction "correct translations" / "all translations"
    """
    assert isinstance(predicted_words, list)
    assert isinstance(true_words, list)
    assert len(predicted_words) == len(true_words)

    num_words = len(predicted_words)
    num_correct = 0

    for prediction, truth in zip(predicted_words, true_words):
        prediction_word = character_list_to_string(prediction)
        truth_word = character_list_to_string(truth)

        if prediction_word == truth_word:
            num_correct += 1

    return float(num_correct) / float(num_words)


def edit_distance(predicted_words, true_words, mode="avg"):
    """
    Calculate the average or median (depending on the value of the `mode` parameter) of the edit distance (Levenshtein
    distance). The library [editdistance](https://github.com/aflc/editdistance) is used to calculate the edit distance.

    Args:
        predicted_words (`list` of `list` of str or `list` of `list` of unicode): A list of predicted words represented
            as lists of characters.
        true_words (`list` of `list` of str or `list` of `list` of unicode): A list of true words represented as lists
            of characters.
        mode (str, optional): How to combine the edit distances of the words. Valid options are "avg" and "median".
             Defaults to "avg".

    Returns:
        float: The average/median edit distance of all words
    """
    assert isinstance(predicted_words, list)
    assert isinstance(true_words, list)
    assert len(predicted_words) == len(true_words)
    assert mode in ["avg", "median"]

    num_words = len(predicted_words)
    edit_distances = []

    for prediction, truth in zip(predicted_words, true_words):
        prediction_word = character_list_to_string(prediction)
        truth_word = character_list_to_string(truth)

        edit_distances.append(editdistance.eval(prediction_word, truth_word))

    if mode == "avg":
        return sum(edit_distances) / float(num_words)
    elif mode == "median":
        return median(edit_distances)
