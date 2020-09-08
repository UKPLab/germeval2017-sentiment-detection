"""Configuration for a whole experiment"""

import pickle as pkl
import logging
import numpy as np
from os import path
from functools import reduce

from ruamel import yaml

from .BaseConfig import BaseConfig
from .CharacterLevelInformationConfig import CharacterLevelInformationConfig
from .EarlyStoppingConfig import EarlyStoppingConfig
from .EmbeddingsConfig import EmbeddingsConfig
from .FileConfig import FileConfig
from .HiddenLayerConfig import HiddenLayerConfig
from .TaskConfig import TaskConfig
from .TrainingConfig import TrainingConfig
from ..constants import CONLL, CLASSIFIER_SOFTMAX, RNN_UNIT_TYPE_LSTM, \
    RNN_UNIT_TYPE_GRU, RNN_UNIT_TYPE_SIMPLE, OPTIMIZER_ADAM, TOKEN_PADDING, TOKEN_UNKNOWN, TOKEN_DATE, \
    TOKEN_TIME, TOKEN_NUMBER, ENCODING_NONE, METRIC_F1, CHAR_LSTM, ACTIVATION_RELU, METRIC_ACCURACY, METRIC_RECALL, \
    METRIC_PRECISION, TASK_TYPE_GENERIC, VALID_METRICS
from ..data.preprocess import merge_embeddings, word_normalize


class ExperimentConfig(BaseConfig):
    """Experiment configuration

    This is the main class for all configurations options of a single experiment.
    """

    def __init__(self, path_to_config):
        """Initialize the experiment configuration.

        Args:
            path_to_config (str): Path to the YAML file that contains the configuration
        """

        assert isinstance(path_to_config, str) \
            and path.exists(path_to_config) \
            and path.isfile(path_to_config)

        self._path_to_config = path_to_config
        self._paths = {}
        self._paths_set = False
        self._prepared = False

        # Default parameters
        self._name = "some-experiment"

        # Training-related
        self._num_runs = 10
        self._epochs = 1
        self._batch_size = 32
        self._training = None

        # RNN related
        self._rnn_unit = RNN_UNIT_TYPE_LSTM
        self._rnn_dropout_input_keep_probability = 1.0
        self._rnn_dropout_output_keep_probability = 1.0
        self._rnn_dropout_state_keep_probability = 1.0
        self._use_bias = True
        self._units = 100
        self._use_variational_dropout = True

        # Architecture-related
        self._tasks = []
        self._short_cut_connections = False

        # Vocabulary-related
        self._word_dropout_keep_probability = 1.0
        self._embedding_size = 100
        self._vocab_size = 0
        self._word2idx = None
        self._char2idx = None
        self._embeddings = []
        self._embedding_weights = None
        self._early_stopping = None
        self._character_level_information = None

        # Evaluation related
        self._eval_metrics = [METRIC_ACCURACY, METRIC_F1, METRIC_PRECISION, METRIC_RECALL]

    def _read_experiment(self, config):
        """
        Read all configuration options related to the experiment itself.
        NOTE: mutates instance properties.

        Args:
            config (dict): Configuration object

        Returns:
            None
        """
        self._name = config.get("name", self._name)
        self._num_runs = config.get("num_runs", self._num_runs)
        self._epochs = config.get("epochs", self._epochs)
        self._batch_size = config.get("batch_size", self._batch_size)
        self._rnn_unit = config.get("rnn_unit", self._rnn_unit)
        self._rnn_dropout_input_keep_probability = config.get(
            "rnn_dropout_input_keep_probability",
            self._rnn_dropout_input_keep_probability
        )
        self._rnn_dropout_output_keep_probability = config.get(
            "rnn_dropout_output_keep_probability",
            self._rnn_dropout_output_keep_probability
        )
        self._rnn_dropout_state_keep_probability = config.get(
            "rnn_dropout_state_keep_probability",
            self._rnn_dropout_state_keep_probability
        )
        self._use_bias = config.get("use_bias", self._use_bias)
        self._use_variational_dropout = config.get("use_variational_dropout", self._use_variational_dropout)
        self._units = config.get("units", self._units)
        self._short_cut_connections = config.get("short_cut_connections", self._short_cut_connections)
        self._word_dropout_keep_probability = config.get(
            "word_dropout_keep_probability", self._word_dropout_keep_probability
        )
        self._eval_metrics = config.get("eval_metrics", self._eval_metrics)

        self._tasks = [
            self._read_task(task_config, index)
            for index, task_config
            in enumerate(config.get("tasks", []))
        ]

        self._embeddings = [
            self._read_embeddings_info(embeddings_config)
            for embeddings_config
            in config.get("embeddings", [])
        ]

        # Read embedding size if no pre-trained embeddings have been specified
        if len(self._embeddings) == 0:
            self._embedding_size = config.get("embedding_size", self._embedding_size)

        self._early_stopping = self._read_early_stopping(config.get("early_stopping", self._early_stopping))
        self._character_level_information = self._read_character_level_information(
            config.get("character_level_information", self._character_level_information)
        )
        self._training = self._read_training(
            config.get("training", None)
        )

    def _read_task(self, task_config, index):
        """
        Read all configuration options related to the tasks.

        Args:
            task_config (dict): Task configuration object

        Returns:
            A task configuration object
        """
        task_number = index + 1

        name = task_config.get("name", "Task_%i" % task_number)
        # Train file is necessary --> raise an error if is not supplied
        train_file = self._read_file_info(task_config["train_file"])
        output_layer = task_config.get("output_layer", 1)
        dev_file = task_config.get("dev_file", None)
        # noinspection PyTypeChecker
        dev_file = None if dev_file is None else self._read_file_info(dev_file)
        test_file = task_config.get("test_file", None)
        # noinspection PyTypeChecker
        test_file = None if test_file is None else self._read_file_info(test_file)
        hidden_layers = [
            self._read_hidden_layer(hidden_layer_config)
            for hidden_layer_config
            in task_config.get("hidden_layers", [])
        ]
        loss = task_config.get("loss", "categorical_crossentropy")
        loss_weight = task_config.get("loss_weight", 1.)
        # Task eval metrics are empty by default
        eval_metrics = task_config.get("eval_metrics", [])
        classifier = task_config.get("classifier", CLASSIFIER_SOFTMAX)
        data_format = task_config.get("data_format", CONLL)
        dropout_keep_probability = task_config.get("dropout_keep_probability", 1.0)
        use_bias = task_config.get("use_bias", True)
        encoding = task_config.get("encoding", ENCODING_NONE)
        task_type = task_config.get("type", TASK_TYPE_GENERIC)

        return TaskConfig(
            name=name,
            train_file=train_file,
            output_layer=output_layer,
            dev_file=dev_file,
            test_file=test_file,
            hidden_layers=hidden_layers,
            loss=loss,
            loss_weight=loss_weight,
            eval_metrics=eval_metrics,
            classifier=classifier,
            data_format=data_format,
            dropout_keep_probability=dropout_keep_probability,
            use_bias=use_bias,
            encoding=encoding,
            type=task_type,
        )

    @staticmethod
    def _read_file_info(file_config):
        """
        Read the configuration options for a file.

        Args:
            file_config (dict): configuration object

        Returns:
            A file configuration object
        """
        # Path is required --> raise an error if it is not available
        path = file_config["path"]
        scheme = file_config.get("scheme", "IOB")
        word_column = file_config.get("word_column", 0)
        label_column = file_config.get("label_column", 1)
        column_separator = file_config.get("column_separator", "tab")
        encoding = file_config.get("encoding", "utf8")

        return FileConfig(
            path=path,
            scheme=scheme,
            word_column=word_column,
            label_column=label_column,
            column_separator=column_separator,
            encoding=encoding
        )

    @staticmethod
    def _read_hidden_layer(hidden_layer_config):
        """
        Read all configuration options related to the hidden layers.

        Args:
            hidden_layer_config (dict): Configuration object

        Returns:
            A hidden layer configuration object
        """
        units = hidden_layer_config.get("units", 100)
        activation = hidden_layer_config.get("activation", ACTIVATION_RELU)
        dropout_keep_probability = hidden_layer_config.get("dropout_keep_probability", 1.0)
        use_bias = hidden_layer_config.get("use_bias", True)

        return HiddenLayerConfig(
            units=units,
            activation=activation,
            use_bias=use_bias,
            dropout_keep_probability=dropout_keep_probability
        )

    @staticmethod
    def _read_embeddings_info(embeddings_config):
        """
        Read all configuration options related to an embeddings file.

        Args:
            embeddings_config (dict or None): Configuration object

        Returns:
            A embeddings configuration object
        """
        # If the user does not specify an embeddings configuration, the network will use randomly initialized embeddings
        if embeddings_config is None:
            return None

        # Path is required --> raise an error if it is not available
        path = embeddings_config["path"]
        lower = embeddings_config.get("lower", False)
        separator = embeddings_config.get("separator", " ")
        encoding = embeddings_config.get("encoding", "utf8")
        size = embeddings_config.get("size")
        gzip = embeddings_config.get("gzip", True)

        return EmbeddingsConfig(
            path=path,
            lower=lower,
            separator=separator,
            encoding=encoding,
            size=size,
            gzip=gzip
        )

    @staticmethod
    def _read_early_stopping(early_stopping_config):
        """
        Read the early stopping configuration options.
        If no options are provided, None is returend.

        Args:
            early_stopping_config (dict or None): Configuration object

        Returns:
            An early stopping configuration object or None
        """
        if early_stopping_config is None:
            return None

        # Task name is required --> raise an error if it is not available
        task_name = early_stopping_config["task_name"]
        metric = early_stopping_config.get("metric", METRIC_F1)
        patience = early_stopping_config.get("patience", 5)

        return EarlyStoppingConfig(
            task_name,
            metric=metric,
            patience=patience
        )

    @staticmethod
    def _read_character_level_information(character_level_information_config):
        """
        Read the character level information configuration options.
        If no options are provided, None is returned.

        Args:
            character_level_information_config (dict or None): Configuration object

        Returns:
            A character level information configuration object or None
        """
        if character_level_information_config is None:
            return None

        network_type = character_level_information_config.get("network_type", CHAR_LSTM)
        dimensionality = character_level_information_config.get("dimensionality", 100)
        hidden_units = character_level_information_config.get("hidden_units", 100)

        return CharacterLevelInformationConfig(
            network_type,
            dimensionality,
            hidden_units
        )

    @staticmethod
    def _read_training(training_config):
        """
        Read the training configuration options.
        If no options are provided, the default configuration is returned.

        Args:
            training_config (dict or None): Configuration object

        Returns:
            TrainingConfig: a training configuration object
        """
        if training_config is None:
            return TrainingConfig(
                optimizer=OPTIMIZER_ADAM,
                optimizer_params={},
                use_gradient_clipping=True,
                clip_norm=5.0
            )

        optimizer = training_config.get("optimizer", OPTIMIZER_ADAM)
        optimizer_params = training_config.get("optimizer_params", {})
        use_gradient_clipping = training_config.get("use_gradient_clipping", True)
        clip_norm = training_config.get("clip_norm", 5.0)

        return TrainingConfig(
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            use_gradient_clipping=use_gradient_clipping,
            clip_norm=clip_norm,
        )

    def read(self):
        """
        Read the configuration from the file provided in the constructor.

        Returns:
            None
        """
        with open(self._path_to_config, mode="r") as stream:
            config = yaml.load(stream, Loader=yaml.Loader)

            self._read_experiment(config)

    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        assert self.paths_set

        logger = logging.getLogger("shared.experiment_config.prepare")
        logger.debug("Preparing experiment config for experiment '%s'", self.name)

        result = True
        words = []
        # First, prepare all tasks
        for i, task in enumerate(self.tasks):
            logger.debug("Preparing task %d...", (i + 1))
            result = result and task.prepare()
            words += task.data_reader.get_words()

        unique_words = set(words)

        embeddings_pkl = path.join(self._paths["pkl"], "embeddings.pkl")
        word2idx_pkl = path.join(self._paths["pkl"], "word2idx.pkl")
        char2idx_pkl = path.join(self._paths["pkl"], "char2idx.pkl")

        if self.character_level_information:
            if path.isfile(char2idx_pkl):
                logger.info("Loading char2idx from pickle file.")
                with open(char2idx_pkl, 'rb') as f:
                    self._char2idx = pkl.load(f)
            else:
                logger.info("Creating char2idx")
                assert self._char2idx is None

                # Get all unique characters
                unique_characters = set()
                for word in unique_words:
                    unique_characters.update(word)

                # Add padding as 0
                unique_characters = [TOKEN_PADDING] + list(unique_characters)

                logger.debug(
                    "There are %d unique characters across all data files of all tasks",
                    len(unique_characters)
                )
                self._char2idx = {k: v for v, k in enumerate(unique_characters)}

        if len(self.embeddings) != 0:
            if path.isfile(embeddings_pkl) and path.isfile(word2idx_pkl):
                logger.info("Loading embedding weights and word2idx from pickle files.")
                with open(embeddings_pkl, 'rb') as f:
                    self._embedding_weights = pkl.load(f)

                with open(word2idx_pkl, 'rb') as f:
                    self._word2idx = pkl.load(f)
            else:
                logger.info("Creating embedding weights and word2idx")

                # word2idx should not be modified yet
                assert self._word2idx is None
                # embedding weights should not be modified yet
                assert self._embedding_weights is None
                self._embedding_weights = []

                result = result and all([embeddings_config.prepare() for embeddings_config in self.embeddings])
                vectors = merge_embeddings(self.embeddings)

                logger.debug("Merged the embeddings from %d files", len(self.embeddings))
                self._embedding_size = sum([embeddings_config.size for embeddings_config in self.embeddings])
                assert len(list(vectors.values())[0]) == self._embedding_size
                logger.debug("Final embeddings have size of %d", self._embedding_size)

                # TODO: make configuarable
                # Reduce size of pre-trained word embeddings based on the words that occur in the documents.
                # Adapted from NR

                needed_vocab = {}
                for word in unique_words:
                    needed_vocab[word] = True
                    needed_vocab[word.lower()] = True
                    needed_vocab[word_normalize(word)] = True

                # Add special tokens
                logger.debug("Adding special tokens")
                self._word2idx = {TOKEN_PADDING: 0}
                self._embedding_weights.append(np.zeros(self._embedding_size))

                for special_token in [TOKEN_UNKNOWN, TOKEN_DATE, TOKEN_TIME, TOKEN_NUMBER]:
                    self._word2idx[special_token] = len(self._word2idx)
                    # TODO: check out other initialization styles
                    self._embedding_weights.append(np.random.uniform(-0.25, 0.25, self._embedding_size))

                for word, vector in vectors.items():
                    if word not in self._word2idx and word in needed_vocab:
                        self._embedding_weights.append(vector)
                        self._word2idx[word] = len(self._word2idx)

                self._embedding_weights = np.asarray(self._embedding_weights)
                self._vocab_size = self._embedding_weights.shape[0]

                logger.info("Loaded embeddings for %d words", self._vocab_size)
                with open(embeddings_pkl, 'wb') as f:
                    pkl.dump(self._embedding_weights, f, -1)
                logger.info("Embeddings file saved as: %s", embeddings_pkl)
                with open(word2idx_pkl, 'wb') as f:
                    pkl.dump(self._word2idx, f, -1)
                logger.info("Word2idx file saved as: %s", word2idx_pkl)

        else:
            if path.isfile(word2idx_pkl):
                with open(word2idx_pkl, 'rb') as f:
                    self._word2idx = pkl.load(f)

            self._vocab_size = len(unique_words)
            # Build "global" word2idx (and ensure that <MASK> is the first element)
            unique_words = [TOKEN_PADDING, TOKEN_UNKNOWN, TOKEN_DATE, TOKEN_TIME, TOKEN_NUMBER] + \
                           [
                               word
                               for word
                               in unique_words
                               if word not in [TOKEN_PADDING, TOKEN_UNKNOWN, TOKEN_DATE, TOKEN_TIME, TOKEN_NUMBER]
                           ]
            self._word2idx = {k: v for v, k in enumerate(unique_words)}

            with open(word2idx_pkl, 'wb') as f:
                pkl.dump(self._word2idx, f, -1)
            logger.info("Word2idx file saved as: %s", word2idx_pkl)

        logger.debug("Word2idx contains %d entries.", len(self._word2idx))

        self._prepared = result
        return result

    def sanity_check(self):
        """
        Validate the configuration (recursively).
        NOTE: run `prepare` before because the configuration is likely not to be valid otherwise.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        logger = logging.getLogger("shared.experiment_config.sanity_check")

        word_dropout_valid = 1.0 >= self.word_dropout_keep_probability >= 0.0

        if not word_dropout_valid:
            logger.warn("Word rnn_dropout_output_keep_probability of %f is not valid. Must be in interval [0.0, 1.0]", self.word_dropout_keep_probability)

        eval_metrics_valid = all([metric in VALID_METRICS for metric in self.eval_metrics])

        if not eval_metrics_valid:
            logger.warn("Some evaluation metric is invalid. Valid metrics are %s", VALID_METRICS)

        rnn_settings_valid = self.rnn_unit in [RNN_UNIT_TYPE_SIMPLE, RNN_UNIT_TYPE_GRU, RNN_UNIT_TYPE_LSTM] \
            and 1.0 >= self.rnn_dropout_input_keep_probability >= 0.0 \
            and 1.0 >= self.rnn_dropout_output_keep_probability >= 0.0 \
            and 1.0 >= self.rnn_dropout_state_keep_probability >= 0.0

        if not rnn_settings_valid:
            logger.warn(
                "RNN settings are invalid. Settings: unit=%s, rnn_dropout_output_keep_probability=%f, "
                "rnn_dropout_state_keep_probability=%f",
                self.rnn_unit,
                self.rnn_dropout_input_keep_probability,
                self.rnn_dropout_output_keep_probability,
                self.rnn_dropout_state_keep_probability
            )

        tasks_valid = reduce(
            lambda x, y: x and y,
            [task.sanity_check() for task in self._tasks],
            True
        )
        if not tasks_valid:
            logger.warn("Some task is invalid")

        early_stopping_valid = self._early_stopping is None or (
            self._early_stopping.sanity_check() and
            # Check if the reference to a task is correct
            self._early_stopping.task_name in [task.name for task in self._tasks]
        )

        if not early_stopping_valid:
            logger.warn("The early stopping configuration is invalid.")

        return all([
            word_dropout_valid,
            eval_metrics_valid,
            rnn_settings_valid,
            tasks_valid,
            early_stopping_valid,
            self.training.sanity_check(),
        ])

    def to_dict(self):
        return {
            "name": self.name,
            "num_runs": self.num_runs,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "units": self.units,
            "short_cut_connections": self.short_cut_connections,
            "rnn_unit": self.rnn_unit,
            "rnn_dropout_input_keep_probability": self.rnn_dropout_input_keep_probability,
            "rnn_dropout_output_keep_probability": self.rnn_dropout_output_keep_probability,
            "rnn_dropout_state_keep_probability": self.rnn_dropout_state_keep_probability,
            "use_bias": self.use_bias,
            "use_variational_dropout": self.use_variational_dropout,
            "tasks": [task.to_dict() for task in self.tasks],
            "embedding_size": self.embedding_size,
            "embeddings": [embeddings.to_dict() for embeddings in self.embeddings],
            "vocab_size": self.vocab_size,
            "word_dropout_keep_probability": self.word_dropout_keep_probability,
            "eval_metrics": self.eval_metrics,
            "early_stopping": self.early_stopping.to_dict() if self.early_stopping is not None else None,
            "character_level_information":
                self.character_level_information.to_dict()
                if self.character_level_information is not None
                else None,
            "training": self.training.to_dict()
        }

    def set_paths(self, paths):
        """
        Set the paths for the experiment.
        Args:
            paths (`dict` of str): Necessary paths.
        """
        self._paths = paths

        for task in self.tasks:
            task.set_paths(paths)

        for embedding in self.embeddings:
            embedding.set_paths(paths)

        self._paths_set = True

    def __str__(self):
        return yaml.dump(self.to_dict())

# <editor-fold desc="Properties">
    @property
    def name(self):
        """

        Returns:
            str: Name of the experiment
        """
        return self._name

    @property
    def num_runs(self):
        """

        Returns:
            int: Number of runs for the experiment
        """
        return self._num_runs

    @property
    def epochs(self):
        """

        Returns:
            int: number of epochs
        """
        return self._epochs

    @property
    def batch_size(self):
        """

        Returns:
            int: batch size
        """
        return self._batch_size

    @property
    def units(self):
        """

        Returns:
            int: number of units in the RNN layers
        """
        return self._units

    @property
    def short_cut_connections(self):
        """
        Short-cut connections as in "A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks" by
        Hashimoto et al. (2017)

        Returns:
            bool: whether or not to use short-cut connections
        """
        return self._short_cut_connections

    @property
    def rnn_unit(self):
        """

        Returns:
            str: RNN unit (either SimpleRNN, GRU or LSTM)
        """
        return self._rnn_unit

    @property
    def rnn_dropout_input_keep_probability(self):
        """

        Returns:
            float: Dropout used by the RNN units
        """
        return self._rnn_dropout_input_keep_probability

    @property
    def rnn_dropout_output_keep_probability(self):
        """

        Returns:
            float: Dropout used by the RNN units
        """
        return self._rnn_dropout_output_keep_probability

    @property
    def rnn_dropout_state_keep_probability(self):
        """

        Returns:
            float: Dropout used between the hidden units (only for GRU and LSTM)
        """
        return self._rnn_dropout_state_keep_probability

    @property
    def use_bias(self):
        """

        Returns:
            bool: Whether or not the RNN units use a bias vector
        """
        return self._use_bias

    @property
    def use_variational_dropout(self):
        """
        Variational dropout as defined in https://arxiv.org/pdf/1512.05287.pdf

        Returns:
            bool: Whether or not to use variational dropout.
        """
        return self._use_variational_dropout

    @property
    def tasks(self):
        """

        Returns:
            `list` of TaskConfig: tasks
        """
        return self._tasks

    @property
    def embedding_size(self):
        """

        Returns:
            int: embedding size
        """
        return self._embedding_size

    @property
    def embeddings(self):
        """

        Returns:
            `list` of EmbeddingsConfig: embeddings configurations
        """
        return self._embeddings

    @property
    def embedding_weights(self):
        """

        Returns:
            np.ndarray: embedding weights as a matrix
        """
        return self._embedding_weights

    @property
    def vocab_size(self):
        """

        Returns:
            int: vocabulary size
        """
        return self._vocab_size

    @property
    def word_dropout_keep_probability(self):
        """

        Returns:
            float: word dropout keep probability rate in [0.0, 1.0]
        """
        return self._word_dropout_keep_probability

    @property
    def eval_metrics(self):
        """

        Returns:
            `list` of str: A list of evaluation metric names
        """
        return self._eval_metrics

    @property
    def early_stopping(self):
        """

        Returns:
            EarlyStoppingConfig: early stopping configuration object
        """
        return self._early_stopping

    @property
    def character_level_information(self):
        """

        Returns:
            CharacterLevelInformationConfig: character level information configuration object
        """
        return self._character_level_information

    @property
    def training(self):
        """

        Returns:
            TrainingConfig: training configuration object
        """
        return self._training

    @property
    def word2idx(self):
        """

        Returns:
             `dict` of int: a mapping from words to indices (shared across all tasks)
        """
        return self._word2idx

    @property
    def char2idx(self):
        """

        Returns:
             `dict` of int: a mapping from characters to indices (shared across all tasks)
        """
        return self._char2idx

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
# </editor-fold>
