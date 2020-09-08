"""
Class to represent the results of a prediction.
"""
import codecs
import logging
import os
import warnings
from numpy import ndarray

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import \
    confusion_matrix, \
    recall_score, \
    precision_score, \
    f1_score, \
    accuracy_score
from tabulate import tabulate

from .argmin_components import evaluate_argmin_components
from .argmin_post_processing import relative_2_absolute
from .argmin_relations import evaluate_argmin_relations
from .metrics import compute_f1, compute_precision, compute_recall, pre_process
from .seq_2_seq_metrics import word_accuracy, edit_distance
from ..config.TaskConfig import TaskConfig
from ..constants import ENCODING_NONE, METRIC_ACCURACY, METRIC_F1, METRIC_PRECISION, METRIC_RECALL, TASK_TYPE_AM, \
    METRIC_WORD_ACCURACY, METRIC_F1_O, METRIC_F1_B, \
    METRIC_PRECISION_O, METRIC_PRECISION_B, METRIC_RECALL_O, METRIC_RECALL_B, METRIC_AM_COMPONENTS_05, \
    METRIC_AM_COMPONENTS_0999, METRIC_AM_RELATIONS_05, METRIC_AM_RELATIONS_0999, METRIC_AVG_EDIT_DISTANCE, \
    METRIC_MEDIAN_EDIT_DISTANCE
from ..data.Sample import Sample
from ..util import swap_dict

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class ResultList(list):
    """
    Class to represent the results of a prediction.
    """

    def __init__(self, result_tuples, label_2_idx, task=None):
        """
        Initialize a result list.
        Creates swapped mapping functions and populates the internal list.
        The list contains tuples with the following entries:
            * Sentence with actual tokens
            * Predicted labels as strings
            * Gold labels as strings
            * Sentence with indices
            * Predicted labels with indices
            * Gold labels with indices
            * Sample object

        Args:
            result_tuples (`list` of `tuple` of object): A list of results represented as tuples consisting of
                (sentence, gold label, predicted label, sample object). The sample object can be used to restore the
                original sentence (words).
            label_2_idx (`dict` of int): A mapping from label names to indices.
            task (TaskConfig): The task to which the results belong to
        """
        assert isinstance(label_2_idx, dict)
        assert isinstance(task, TaskConfig)

        logger = logging.getLogger("shared.result_list.init")

        list.__init__(self)

        self.label_2_idx = label_2_idx
        self.idx_2_label = swap_dict(label_2_idx)
        self.task = task

        logger.debug("Initializing a result list for %d sentences", len (result_tuples))
        for sentence, gold_labels, predicted_labels, sample in result_tuples:
            assert isinstance(sample, Sample)
            assert len(sentence) == len(gold_labels) == len(predicted_labels)

            word_sentence = sample.raw_tokens
            word_gold_labels = sample.raw_labels
            docid = sample.docid
            word_predicted_labels = [self.idx_2_label[idx] for idx in predicted_labels]

            # Removal of padding if necessary
            if len(word_sentence) != len(sentence):
                # logger.debug("There is a padded sentence. Remove padding.")
                # The raw sentence as stored in the sample object has the true length
                true_length = len(word_sentence)
                sentence = sentence[:true_length]
                gold_labels = gold_labels[:true_length]
                predicted_labels = predicted_labels[:true_length]

            self.append((
                word_sentence,
                word_predicted_labels,
                word_gold_labels,
                sentence,
                predicted_labels,
                gold_labels,
                sample
            ))

    def get_true_and_pred(self):
        """
        From the unmasked data in the result list, create a list of predictions and a list of truths.
        Returns:
            `tuple` of `list` of str: A tuple consisting of the truths and the predictions (in this order).
        """
        y_true = []
        y_pred = []

        for _, pred, gold, _, _, _, sample in self:
            for pred_label, gold_label in zip(pred, gold):
                y_true.append(gold_label)
                y_pred.append(pred_label)

        return y_true, y_pred

    def get_true_and_pred_sentences(self, word=False):
        """
        Retrieve all true and predicted sentence labels. If `word` is True, retrieve the word representation for labels.
        Otherwise, retrieve the index representation. The latter is required for calculating metrics on BIO.

        Args:
            word (bool): Whether to use word or index representations for the labels.

        Returns:
            `tuple` of `list` of `list` of str or `tuple` of `list` of `list` of int: A tuple consisting of gold label
                sentences and predictions (in this order).
        """

        true_sentences = []
        predicted_sentences = []

        for entry in self:
            if word:
                predicted_sentences.append(entry[1])
                true_sentences.append(entry[2])
            else:
                predicted_sentences.append(entry[4])
                true_sentences.append(entry[5])

        return true_sentences, predicted_sentences

    def confusion_matrix(self):
        """
        Compute the confusion matrix for the result list.
        Returns:
            Confusion matrix
        """
        y_true, y_pred = self.get_true_and_pred()

        return confusion_matrix(y_true, y_pred, labels=list(self.idx_2_label.values()))

    def print_confusion_matrix(self, matrix=None):
        """
        Generate a ASCII representation for the confusion matrix.
        Args:
            matrix: A confusion matrix.

        Returns:
            A well-formatted confusion matrix.
        """
        if matrix is None:
            matrix = self.confusion_matrix()

        if isinstance(matrix, ndarray):
            matrix = matrix.tolist()

        labels = list(self.idx_2_label.values())
        for row_idx in range(len(matrix)):
            # Prepend label for rows
            matrix[row_idx] = [labels[row_idx]] + matrix[row_idx]

        print (tabulate(matrix, headers=labels))

    def accuracy(self):
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.accuracy_score

        Returns:
                float: accuracy score
        """
        y_true, y_pred = self.get_true_and_pred()

        return accuracy_score(y_true, y_pred)

    def precision(self, correct_bio_errors="No"):
        """
        Calculate the precision. If the task uses BIO, IOB or IOBES encoding, a special calculation method is used.
        Otherwise, we fall back to the scikit learn implementation.
        Args:
             correct_bio_errors (str): If this is set to "O" or "B", a correction of incorrect "I-" labels is performed.
                See `metrics.py` for further details.
        Returns:
            float: precision score
        """

        if self.task is None or self.task.encoding == ENCODING_NONE:
            # Not BIO, IOB or IOBES
            y_true, y_pred = self.get_true_and_pred()
            return precision_score(y_true, y_pred, labels=list(self.idx_2_label.values()), average="macro")
        else:
            y_true, y_pred = self.get_true_and_pred_sentences(word=False)
            y_true, y_pred = pre_process(
                y_pred,
                y_true,
                self.idx_2_label,
                correct_bio_errors=correct_bio_errors,
                encoding_scheme=self.task.encoding
            )
            return compute_precision(y_pred, y_true)

    def recall(self, correct_bio_errors="No"):
        """
        Calculate the precision. If the task uses BIO, IOB or IOBES encoding, a special calculation method is used.
        Otherwise, we fall back to the scikit learn implementation.
        Args:
             correct_bio_errors (str): If this is set to "O" or "B", a correction of incorrect "I-" labels is performed.
                See `metrics.py` for further details.
        Returns:
            float: precision score
        """

        if self.task is None or self.task.encoding == ENCODING_NONE:
            # Not BIO, IOB or IOBES
            y_true, y_pred = self.get_true_and_pred()
            return recall_score(y_true, y_pred, labels=list(self.idx_2_label.values()), average="macro")
        else:
            y_true, y_pred = self.get_true_and_pred_sentences(word=False)
            y_true, y_pred = pre_process(
                y_pred,
                y_true,
                self.idx_2_label,
                correct_bio_errors=correct_bio_errors,
                encoding_scheme=self.task.encoding
            )
            return compute_recall(y_pred, y_true)

    def f1(self, correct_bio_errors="No"):
        """
        Calculate the precision. If the task uses BIO, IOB or IOBES encoding, a special calculation method is used.
        Otherwise, we fall back to the scikit learn implementation.
        Args:
             correct_bio_errors (str): If this is set to "O" or "B", a correction of incorrect "I-" labels is performed.
                See `metrics.py` for further details.
        Returns:
            float: precision score
        """

        if self.task is None or self.task.encoding == ENCODING_NONE:
            # Not BIO, IOB or IOBES
            y_true, y_pred = self.get_true_and_pred()
            return f1_score(y_true, y_pred, labels=list(self.idx_2_label.values()), average="macro")
        else:
            y_true, y_pred = self.get_true_and_pred_sentences(word=False)
            y_true, y_pred = pre_process(
                y_pred,
                y_true,
                self.idx_2_label,
                correct_bio_errors=correct_bio_errors,
                encoding_scheme=self.task.encoding
            )
            return compute_f1(y_pred, y_true)

    def argmin_components(self, ratio=0.5):
        """
        Calculate the AM components score at the specified ratio.

        Args:
            ratio (float): Ratio for score calculation.

        Returns:
            float: f1 score
        """
        conll_list = self.as_conll_list()
        prediction_list = relative_2_absolute(conll_list, 0, 2)
        truth_list = relative_2_absolute(conll_list, 0, 1)
        result = evaluate_argmin_components(prediction_list, truth_list, 2, 2, ratio=ratio)
        return result[3]

    def argmin_relations(self, ratio=0.5):
        """
        Calculate the AM relations score at the specified ratio.

        Args:
            ratio (float): Ratio for score calculation.

        Returns:
            float: f1 score
        """
        conll_list = self.as_conll_list()
        prediction_list = relative_2_absolute(conll_list, 0, 2)
        truth_list = relative_2_absolute(conll_list, 0, 1)
        result = evaluate_argmin_relations(prediction_list, truth_list, 2, 2, ratio=ratio)
        return result[3]

    def word_accuracy(self):
        """
        Calculate the word accuracy.
        Use this only for seq2seq tasks.

        Returns:
            float: word accuracy
        """
        y_true, y_pred = self.get_true_and_pred_sentences(word=True)
        return word_accuracy(y_pred, y_true)

    def edit_distance(self, mode="avg"):
        """
        Calculate the edit distance.
        Use this only for seq2seq tasks.

        Args:
            mode (str, optional): How to combine the edit distances of the words. Valid options are "avg" and "median".
             Defaults to "avg".

        Returns:
            float: average edit distance
        """
        assert mode in ["avg", "median"]

        y_true, y_pred = self.get_true_and_pred_sentences(word=True)
        return edit_distance(y_pred, y_true, mode)

    def compute_metric_by_name(self, metric_name):
        """
        Compute the metric identified by `metric_name`. If the metric name is unknown,
        a value error is raised.

        Args:
            metric_name (str): The name of a metric.

        Returns:
            float: metric value
        """
        if metric_name == METRIC_ACCURACY:
            return self.accuracy()
        elif metric_name == METRIC_F1:
            return self.f1()
        elif metric_name == METRIC_F1_O:
            return self.f1(correct_bio_errors="O")
        elif metric_name == METRIC_F1_B:
            return self.f1(correct_bio_errors="B")
        elif metric_name == METRIC_PRECISION:
            return self.precision()
        elif metric_name == METRIC_PRECISION_O:
            return self.precision(correct_bio_errors="O")
        elif metric_name == METRIC_PRECISION_B:
            return self.precision(correct_bio_errors="B")
        elif metric_name == METRIC_RECALL:
            return self.recall()
        elif metric_name == METRIC_RECALL_O:
            return self.recall(correct_bio_errors="O")
        elif metric_name == METRIC_RECALL_B:
            return self.recall(correct_bio_errors="B")
        elif metric_name == METRIC_AM_COMPONENTS_05:
            return self.argmin_components(ratio=0.5)
        elif metric_name == METRIC_AM_COMPONENTS_0999:
            return self.argmin_components(ratio=0.999)
        elif metric_name == METRIC_AM_RELATIONS_05:
            return self.argmin_relations(ratio=0.5)
        elif metric_name == METRIC_AM_RELATIONS_0999:
            return self.argmin_components(ratio=0.999)
        elif metric_name == METRIC_WORD_ACCURACY:
            return self.word_accuracy()
        elif metric_name == METRIC_AVG_EDIT_DISTANCE:
            return self.edit_distance(mode="avg")
        elif metric_name == METRIC_MEDIAN_EDIT_DISTANCE:
            return self.edit_distance(mode="median")
        else:
            raise ValueError("Metric with name %s is not supported by this method." % metric_name)

    def as_conll_list(self, delimiter="\t"):
        """
        Build a document in CoNNL format, but each line is a separate string within
        a list.

        Args:
            delimiter (str, optional): Which character is used as a column separator. Defaults to tab (`\t`).

        Returns:
            `list` of str: A list of lines in CoNLL format (token truth prediction).
        """
        output = []
        for x, y, gold, _, _, _, sample in self:
            #print(sample.docid)
            docid = ""
            if sample.docid != None:
                docid = sample.docid
            output.append(docid)
            for i in range(len(x)):
                output.append(delimiter.join([x[i], gold[i], y[i]]))

            # Add empty line to separate sentences
            output.append("")

        return output

    def __str__(self):
        """
        Build a string representation for an instance of the result list class.
        Returns:
            Data in CONLL format with predicted labels in the last row.
        """
        return "\n".join(self.as_conll_list())

    def predictions_to_file(self, prediction_dir_path, filename):
        """
        Write predictions to a file.

        If the task is AM, two files are written that adhere to the format used by SE and JD.

        Args:
            prediction_dir_path (str): Path to prediction directory.
            filename (str): Prediction filename
        """
        assert os.path.exists(prediction_dir_path), "Expected that prediction directory path exists"
        assert os.path.isdir(prediction_dir_path), "Expected that prediction directory path points to a directory"

        logger = logging.getLogger("shared.result_list.predictions_to_file")
        logger.debug("Writing predictions to file(s)")

        if self.task and self.task.type == TASK_TYPE_AM:
            pred_file_path = os.path.join(prediction_dir_path, filename + ".pred.corr.abs")
            gold_file_path = os.path.join(prediction_dir_path, filename + ".truth.corr.abs")
            logger.debug("Files: %s", [pred_file_path, gold_file_path])

            conll_list = self.as_conll_list()
            prediction_list = relative_2_absolute(conll_list, 0, 2)
            truth_list = relative_2_absolute(conll_list, 0, 1)

            with codecs.open(pred_file_path, mode="w", encoding="utf8") as f:
                f.write("\n".join(prediction_list))

            with codecs.open(gold_file_path, mode="w", encoding="utf8") as f:
                f.write("\n".join(truth_list))
        else:
            file_path = os.path.join(prediction_dir_path, filename)
            logger.debug("File: %s", file_path)

            with codecs.open(file_path, mode="w", encoding="utf8") as f:
                f.write(self.__str__())

    def metrics_as_list(self):
        """
        Provides the performance metrics for the result list as a list (useful for storing in CSV format).
        Entries in the list:
            * Number of performed predictions
            * Number of correct predictions
            * Number of incorrect predictions
            * Accuracy
            * Precision
            * Recall
            * F1 score
        Returns:
            `list` of int or `list` of float: List of metrics
        """
        y_true, y_pred = self.get_true_and_pred()
        num_total = len(y_true)
        num_correct = len([1 for t, p in zip(y_true, y_pred) if t == p])
        num_false = num_total - num_correct

        return [
            num_total,
            num_correct,
            num_false,
            self.accuracy(),
            self.precision(),
            self.recall(),
            self.f1()
        ]
