"""
Computes the F1 score, precision, and recall on BIO tagged data.
Copied from NR's MTL network implementation. Adapted to fit project's needs.
"""
import logging

from ..constants import ENCODING_BIO, ENCODING_IOB, ENCODING_IOBES
from sklearn.metrics import f1_score

def pre_process(predictions, correct, idx_2_label, correct_bio_errors="No", encoding_scheme=ENCODING_BIO):
    """
    Pre-process the index-based label lists so that they are converted to string representations for each
    label.
    NOTE: this method does not mutate the input!
    Indices will be replaced by words using `idx_2_label`. If `correct_bio_errors` is set to either "O"
    or "B", predicted labels will be corrected. For further information see `check_bio_encoding`.
    The provided `encoding_scheme` indicates whether or not it is necessary to convert the encoding.

    Args:
        predictions (`list` of `list` of int):
        correct (`list` of `list` of int):
        idx_2_label (`dict` of str):
        correct_bio_errors (str):
        encoding_scheme (str):

    Returns:

    """
    label_predicted = []
    for sentence in predictions:
        label_predicted.append([idx_2_label[element] for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([idx_2_label[element] for element in sentence])

    encoding_scheme = encoding_scheme.upper()

    if encoding_scheme == ENCODING_IOBES:
        convert_iobes_to_bio(label_predicted)
        convert_iobes_to_bio(label_correct)
    elif encoding_scheme == ENCODING_IOB:
        convert_iob_to_bio(label_predicted)
        convert_iob_to_bio(label_correct)

    check_bio_encoding(label_predicted, correct_bio_errors)

    return label_predicted, label_correct


def compute_f1(label_predicted, label_correct):
    label_truth = [element for sublist in label_correct for element in sublist]
    label_preds = [element for sublist in label_predicted for element in sublist]
    label_dict = set(label_truth+label_preds)
    f1 = f1_score(label_truth, label_preds, list(label_dict),average='macro')
    return f1


def convert_iob_to_bio(data_set):
    """ Convert inplace IOB encoding to BIO encoding """
    for sentence in data_set:
        prev_val = 'O'
        for pos in range(len(sentence)):
            first_char = sentence[pos][0]
            if first_char == 'I':
                if prev_val == 'O' or prev_val[1:] != sentence[pos][1:]:
                    sentence[pos] = 'B' + sentence[pos][1:]  # Change to begin tag

            prev_val = sentence[pos]


def convert_iobes_to_bio(data_set):
    """ Convert inplace IOBES encoding to BIO encoding """
    for sentence in data_set:
        for pos in range(len(sentence)):
            first_char = sentence[pos][0]
            if first_char == 'S':
                sentence[pos] = 'B' + sentence[pos][1:]
            elif first_char == 'E':
                sentence[pos] = 'I' + sentence[pos][1:]


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correct_count = 0
    count = 0

    for sentence_idx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentence_idx]
        correct = correct_sentences[sentence_idx]

        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B':  # A new chunk starts
                count += 1

                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctly_found = True

                    while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctly_found = False

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == 'I':  # The chunk in correct was longer
                            correctly_found = False

                    if correctly_found:
                        correct_count += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correct_count) / count

    return precision


def compute_recall(guessed_sentences, correct_sentences):
    return compute_precision(correct_sentences, guessed_sentences)


def check_bio_encoding(predictions, correct_bio_errors):
    logger = logging.getLogger("shared.metrics.check_bio_encoding")
    errors = 0
    labels = 0

    for sentence_idx in range(len(predictions)):
        label_started = False
        label_class = None

        for label_idx in range(len(predictions[sentence_idx])):
            label = predictions[sentence_idx][label_idx]
            if label.startswith('B-'):
                labels += 1
                label_started = True
                label_class = label[2:]

            elif label == 'O':
                label_started = False
                label_class = None
            elif label.startswith('I-'):
                if not label_started or label[2:] != label_class:
                    errors += 1

                    if correct_bio_errors.upper() == 'B':
                        predictions[sentence_idx][label_idx] = 'B-' + label[2:]
                        label_started = True
                        label_class = label[2:]
                    elif correct_bio_errors.upper() == 'O':
                        predictions[sentence_idx][label_idx] = 'O'
                        label_started = False
                        label_class = None
            else:
                print ( predictions[sentence_idx], label)
                assert False, "Label: %s" % label  # Should never be reached

    if errors > 0:
        labels += errors
        logger.debug("Wrong BIO-Encoding %d/%d labels, %.2f%%" % (errors, labels, errors / float(labels) * 100), )


