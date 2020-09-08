"""Class for a single task configuration"""

import logging
from functools import reduce

from .BaseConfig import BaseConfig
from .FileConfig import FileConfig
from .HiddenLayerConfig import HiddenLayerConfig
from ..constants import DATA_TYPE_TRAIN, DATA_TYPE_DEV, DATA_TYPE_TEST, CLASSIFIER_CRF, CLASSIFIER_SOFTMAX, \
    ENCODING_IOBES, ENCODING_IOB, ENCODING_BIO, ENCODING_NONE, TASK_TYPE_AM, TASK_TYPE_GENERIC, VALID_METRICS
from ..data import DataReaderFactory


class TaskConfig(BaseConfig):
    def __init__(
            self,
            name,
            train_file,
            output_layer,
            dev_file,
            test_file,
            hidden_layers,
            loss,
            loss_weight,
            eval_metrics,
            classifier,
            data_format,
            use_bias,
            dropout_keep_probability,
            encoding,
            type,
    ):
        """Initialize task configuration.

        Args:
            name (str): Name of the task (should not contain spaces or special characters)
            train_file (FileConfig): File object for the training data
            output_layer (int): Output layer associated with this task (as in paper by Sogaard and Goldberg)
            dev_file (FileConfig or None): File object for the dev data
            test_file (FileConfig or None): File object for the test data
            hidden_layers (HiddenLayerConfig or `list` of HiddenLayerConfig): A hidden layer config or a list of hidden
                layer configs to define the hidden layer(s) between the task layer and the classifier.
            loss (str or callable): A loss function name or a loss function itself for this task.
            loss_weight (float): A weight that decides how much the loss for this task influences the parameter updates.
            eval_metrics (`list` of str): One or more metrics as a list of identifiers
            classifier (str): A classifier for this task. Either "softmax" or "CRF"
            data_format (str): data format of the files (usually: CONLL)
            use_bias (bool): Whether or not tu use a bias vector
            dropout_keep_probability (float): Keep probability for dropout_keep_probability. Between 0.0 and 1.0
            encoding (str): Which encoding type is used
            type (str): The task's type. This is used for task-specific pre- or post-processing. Only required if the
                task requires such pre-/post-processing.
        """

        # Ensure that data types are correct
        assert isinstance(name, str)
        assert isinstance(train_file, FileConfig)
        assert isinstance(output_layer, int)
        assert dev_file is None or isinstance(dev_file, FileConfig)
        assert test_file is None or isinstance(test_file, FileConfig)
        assert isinstance(hidden_layers, HiddenLayerConfig) or isinstance(hidden_layers, list)
        assert isinstance(loss, str) or callable(loss)
        assert isinstance(loss_weight, float)
        assert isinstance(eval_metrics, list)
        assert isinstance(classifier, str) and (classifier == CLASSIFIER_CRF or classifier == CLASSIFIER_SOFTMAX)
        assert isinstance(data_format, str)
        assert isinstance(use_bias, bool)
        assert isinstance(dropout_keep_probability, float) and 0.0 <= dropout_keep_probability <= 1.0
        assert isinstance(encoding, str) and encoding in [ENCODING_NONE, ENCODING_BIO, ENCODING_IOB, ENCODING_IOBES]
        assert type in [TASK_TYPE_GENERIC, TASK_TYPE_AM]

        self._name = name
        self._train_file = train_file
        self._dev_file = dev_file
        self._test_file = test_file
        self._output_layer = output_layer
        self._hidden_layers = hidden_layers if isinstance(hidden_layers, list) else [hidden_layers]
        self._loss = loss
        self._loss_weight = loss_weight
        self._eval_metrics = eval_metrics
        self._classifier = classifier
        self._data_format = data_format
        self._use_bias = use_bias
        self._dropout_keep_probability = dropout_keep_probability
        self._encoding = encoding
        self._type = type

        self._data_reader = DataReaderFactory.create_data_reader(data_format)

        self._prepared = False
        self._paths_set = False
        self._paths = {}

    @property
    def name(self):
        """str: Name of the task (should not contain spaces or special characters)"""
        return self._name

    @property
    def train_file(self):
        """FileConfig: File object for the training data"""
        return self._train_file

    @property
    def dev_file(self):
        """int: Output layer associated with this task (as in paper by Sogaard and Goldberg)"""
        return self._dev_file

    @property
    def test_file(self):
        """FileConfig: File object for the dev data"""
        return self._test_file

    @property
    def output_layer(self):
        """FileConfig: File object for the test data"""
        return self._output_layer

    @property
    def hidden_layers(self):
        """`list` of HiddenLayerConfig:

            A hidden layer config or a list of hidden layer configs to define the hidden layer(s)
            between the task layer and the classifier.
        """
        return self._hidden_layers

    @property
    def loss(self):
        """str: callable: A loss function name or a loss function itself for this task."""
        return self._loss

    @property
    def loss_weight(self):
        """float: A weight that decides how much the loss for this task influences the parameter updates."""
        return self._loss_weight

    @property
    def eval_metrics(self):
        """`list` of str: One or more metrics as a list of identifiers"""
        return self._eval_metrics

    @property
    def classifier(self):
        """str: which classifier to use for this task"""
        return self._classifier

    @property
    def data_format(self):
        """str: data format for the files"""
        return self._data_format

    @property
    def use_bias(self):
        """bool: Whether or not tu use a bias vector"""
        return self._use_bias

    @property
    def dropout_keep_probability(self):
        """float: Keep probability for dropout_keep_probability. Between 0.0 and 1.0"""
        return self._dropout_keep_probability

    @property
    def encoding(self):
        """str: Which encoding type is used"""
        return self._encoding

    @property
    def type(self):
        """

        Returns:
            str: The task's type. This is used for task-specific pre- or post-processing. Only required if the
                task requires such pre-/post-processing.
        """
        return self._type

    @property
    def data_reader(self):
        """BaseDataReader: data reader for the task"""
        return self._data_reader

    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        # Prepare files
        result = True
        for file_config in [self.train_file, self.dev_file, self.test_file]:
            result = result and file_config.prepare()

        self._data_reader.add_paths(self._paths)
        self._data_reader.add_name(self._name)

        self._data_reader.add_files({
            DATA_TYPE_TRAIN: self.train_file,
            DATA_TYPE_DEV: self.dev_file,
            DATA_TYPE_TEST: self.test_file
        })

        self._prepared = result
        return result

    def sanity_check(self):
        logger = logging.getLogger("shared.task_config.sanity_check")

        hidden_layers_valid = reduce(
            lambda x, y: x and y,
            [hidden_layer.sanity_check() for hidden_layer in self._hidden_layers],
            True
        )
        if not hidden_layers_valid:
            logger.warn("Some hidden layer is invalid")

        files_valid = \
            self._train_file is not None and \
            self._train_file.sanity_check() and \
            (self._dev_file is None or self._dev_file.sanity_check()) and \
            (self._test_file is None or self.test_file.sanity_check())
        if not files_valid:
            logger.warn("Some files are not valid")

        eval_metrics_valid = all([metric in VALID_METRICS for metric in self.eval_metrics])

        if not eval_metrics_valid:
            logger.warn("Some evaluation metric is invalid. Valid metrics are %s", VALID_METRICS)

        return hidden_layers_valid and files_valid and eval_metrics_valid

    def to_dict(self):
        return {
            "name": self.name,
            "train_file": self.train_file.to_dict(),
            "dev_file": self.dev_file if self.dev_file is None else self.dev_file.to_dict(),
            "test_file": self.test_file if self.test_file is None else self.test_file.to_dict(),
            "output_layer": self.output_layer,
            "hidden_layers": [hidden_layer.to_dict() for hidden_layer in self.hidden_layers],
            "loss": self.loss,
            "loss_weight": self.loss_weight,
            "eval_metrics": self.eval_metrics,
            "data_format": self.data_format,
            "use_bias": self.use_bias,
            "dropout_keep_probability": self.dropout_keep_probability,
            "classifier": self.classifier,
            "type": self.type,
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

        self.train_file.set_paths(paths)
        if self.dev_file:
            self.dev_file.set_paths(paths)
        if self.test_file:
            self.test_file.set_paths(paths)

        for hidden_layer in self.hidden_layers:
            hidden_layer.set_paths(paths)

        self._paths_set = True

    @property
    def paths_set(self):
        """
        Check if the paths have been set. Should be true after `set_paths` has been called.
        Returns:
            bool: True if the paths have been set, False otherwise.
        """
        return self._paths_set
