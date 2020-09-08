"""Tensorflow network for multi-task learning

This is motivated by the sequence tagging model shown in here:
https://github.com/guillaumegenthial/sequence_tagging/blob/master/model.py

Also see the associated blog article:
https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
"""
import logging
import os
import time
import pickle as pkl

import numpy as np
#import tensorflow as tf  # make backward compatible  
import tensorflow as tf
import tensorflow_addons as tfa
tf.compat.v1.disable_eager_execution()

from mappings import ACTIVATION_MAPPING, OPTIMIZER_MAPPING, RNN_CELL_MAPPING
from shared_modules.config.ExperimentConfig import ExperimentConfig
from shared_modules.config.HiddenLayerConfig import HiddenLayerConfig
from shared_modules.config.TaskConfig import TaskConfig
from shared_modules.constants import CLASSIFIER_CRF, DATA_TYPE_TRAIN, DATA_TYPE_DEV, TOKEN_PADDING, PREFIX_MODEL_WEIGHTS, \
    CHAR_CNN, CHAR_LSTM, DIR_TENSOR_BOARD
from shared_modules.data.Batch import Batch
from shared_modules.data.Batches import Batches
from shared_modules.eval.ResultList import ResultList
from shared_modules.network.BaseNeuralNetwork import BaseNeuralNetwork
from shared_modules.util import append_to_csv


class Network(BaseNeuralNetwork):
    """
    Bi-LSTM network that allows to define multiple tasks and choose between a softmax and CRF classifier for each task.
    """
    def __init__(self, config, paths, session_id, run_idx=0):
        """
        Initialize the network.
        Args:
            config (ExperimentConfig): A configuration object for the experiment
            paths (`dict` of str): A dictionary that contains all paths
            session_id (str): Session identifier (UUID-v4)
            run_idx (int, optional): Index of the current run this network is used in (zero-based index)
        """
        assert isinstance(config, ExperimentConfig)
        assert isinstance(paths, dict)
        assert isinstance(session_id, str)
        assert isinstance(run_idx, int) and run_idx >= 0

        self._config = config
        self._paths = paths
        self._session_id = session_id
        self._run_idx = run_idx

        self._init = None
        self._input_word = None
        self._input_characters = None
        self._inputs_label = {}
        self._input_sequence_length = None
        self._input_word_length = None
        self._embeddings_layer = None
        self._shared_layers_output = {}
        self._projections = {}
        self._losses = {}
        self._predictions = {}
        self._transition_params = {}
        self._operations_train = {}
        self._gradient_norms = {}

        self._task_by_name = {
            task.name: task
            for task in self.config.tasks
        }

        logger = logging.getLogger("%s.Network.__init__" % self.config.name)
        logger.debug("Instantiated the network. Require call to `build` to build the computation graph.")

    @property
    def config(self):
        """

        Returns:
            ExperimentConfig: the configuration object for the experiment
        """
        return self._config

    def _build_initialization(self):
        """
        Build the initialization node for the computation graph.
        IMPORTANT: this has to be the last call that modifies the computation graph.
        """
        logger = logging.getLogger("%s.Network._build_initialization" % self.config.name)
        logger.debug("Build global variable initialization node.")
        self._init = tf.compat.v1.global_variables_initializer()

    def _build_placeholders(self):
        """
        Build placeholder nodes, i.e. nodes for the network input.
        """
        logger = logging.getLogger("%s.Network._build_placeholders" % self.config.name)
        logger.debug("Creating placeholders")

        # Placeholder for word inputs
        # shape = (batch size, max length of sentence in batch)
        self._input_word = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="word_indices")

        # Placeholder for sequence length input
        # shape = (batch size)
        self._input_sequence_length = tf.compat.v1.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # Placeholder for label input (separate input for each task)
        # shape = (batch size, max length of sentence in batch)
        self._inputs_label = {
            task.name: tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="label_indices_%s" % task.name)
            for task in self.config.tasks
        }

        # Placeholder for character inputs
        # shape = (batch size, max length of sentence in batch, max length of word in batch)
        self._input_characters = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None], name="character_indices")

        # Placeholder for word lengths
        # shape = (batch size, max length of sentence in batch)
        self._input_word_length = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="word_lengths")

    def _build_character_embeddings_layer(self):
        """
        Build the extractor for character level information.
        Based on https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
        """
        logger = logging.getLogger("%s.Network._build_character_embeddings_layer" % self.config.name)
        logger.debug("Building character embeddings layer")

        with tf.compat.v1.variable_scope("characters"):
            # 1. Get character embeddings
            logger.debug(
                "Building character embeddings layer with shape (%d, %d)",
                len(self.config.char2idx),
                self.config.character_level_information.dimensionality
            )
            embeddings_variable = tf.compat.v1.get_variable(
                name="embeddings_variable",
                dtype=tf.float32,
                shape=[len(self.config.char2idx), self.config.character_level_information.dimensionality]
            )

            character_embeddings = tf.nn.embedding_lookup(
                params=embeddings_variable,
                ids=self._input_characters,
                name="character_embeddings"
            )

            # 2. Put the time dimension on axis=1
            # shape should be (batch size * sequence length, word length, dimensionality of embeddings)
            shape = tf.shape(input=character_embeddings)
            character_embeddings = tf.reshape(
                character_embeddings,
                shape=[-1, shape[-2], self.config.character_level_information.dimensionality]
            )
            word_length = tf.reshape(self._input_word_length, shape=[-1])

            # 3. Select type of network
            if self.config.character_level_information.network_type == CHAR_LSTM:
                logger.debug("Using LSTM to extract character level information (Ma & Hovy, 2016)")
                _, ((_, output_fw), (_, output_bw)) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    tf.compat.v1.nn.rnn_cell.LSTMCell(self.config.character_level_information.hidden_units, state_is_tuple=True),
                    tf.compat.v1.nn.rnn_cell.LSTMCell(self.config.character_level_information.hidden_units, state_is_tuple=True),
                    character_embeddings,
                    sequence_length=word_length,
                    dtype=tf.float32
                )
                output = tf.concat([output_fw, output_bw], axis=-1)
                output = tf.reshape(
                    output,
                    shape=[-1, shape[1], 2 * self.config.character_level_information.hidden_units]
                )
            elif self.config.character_level_information.network_type == CHAR_CNN:
                raise NotImplementedError("CNN extraction for character level information is not implemented yet.")
            else:
                raise ValueError(
                    "Character extraction network type '%s' is not supported." %
                    self.config.character_level_information.network_type
                )

            return output

    def _build_embedding_layer(self):
        """
        Build the embedding layer and attach it to the word input.
        """
        logger = logging.getLogger("%s.Network._build_embedding_layer" % self.config.name)
        logger.debug("Building embeddings layer")
        with tf.compat.v1.variable_scope("words"):
            if self.config.embedding_weights is not None:
                logger.debug("Using pre-trained embeddings")
                embeddings_variable = tf.Variable(
                    self.config.embedding_weights,
                    name="embeddings_variable",
                    dtype=tf.float32,
                    trainable=False
                )
            else:
                logger.debug("Using randomly initialized embeddings")
                embeddings_variable = tf.compat.v1.get_variable(
                    name="embeddings_variable",
                    dtype=tf.float32,
                    # Hardcoded 300 dimensional embeddings if no pre-trained embeddings are used
                    shape=[len(self.config.word2idx), 300]
                )

            self._embeddings_layer = tf.nn.embedding_lookup(
                params=embeddings_variable,
                ids=self._input_word,
                name="word_embeddings"
            )

        if self.config.character_level_information:
            logger.debug("Concatenating word and character-level information")
            self._embeddings_layer = tf.concat(
                [self._embeddings_layer, self._build_character_embeddings_layer()],
                axis=-1
            )

        self._embeddings_layer = tf.nn.dropout(self._embeddings_layer, 1 - (self.config.word_dropout_keep_probability))

    def _build_shared_layers(self):
        """
        Build the shared layers. The number of layers and their size depends on the configuration.
        """
        logger = logging.getLogger("%s.Network._build_shared_layers" % self.config.name)
        logger.debug("Building shared layers")
        input_layer = self._embeddings_layer

        # The maximum output layer index is a 0-based index. Hence, the number of layers is this index + 1.
        num_layers = max([task.output_layer for task in self.config.tasks]) + 1
        logger.debug("There are %d shared layers", num_layers)

        logger.debug("Using %s RNN cells", self.config.rnn_unit)
        RNNCell = RNN_CELL_MAPPING[self.config.rnn_unit]

        if self.config.short_cut_connections:
            logger.debug("Using short-cut connections (Hashimoto et al., 2017)")

        for num in range(num_layers):
            logger.debug("Building %d. shared layer" % (num + 1))

            with tf.compat.v1.variable_scope("shared-layer_%d" % num):
                # Tasks that end on this layer:
                output_here = [task for task in self.config.tasks if task.output_layer == num]

                # Tasks that end on higher layers:
                output_later = [task for task in self.config.tasks if task.output_layer > num]

                logger.debug("Terminate %d tasks here", len(output_here))
                logger.debug("Terminate %d tasks later", len(output_later))

                rnn_cell_fw = RNNCell(self.config.units)
                rnn_cell_bw = RNNCell(self.config.units)

                # Apply dropout
                # Value for input size is the number of features as proposed in
                # https://www.reddit.com/r/tensorflow/comments/6d2d2t/meaning_of_input_size_paramener_for/.
                # The shape tuple always has the number of features in the last element.
                rnn_cell_fw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                    rnn_cell_fw,
                    input_keep_prob=self.config.rnn_dropout_input_keep_probability,
                    output_keep_prob=self.config.rnn_dropout_output_keep_probability,
                    state_keep_prob=self.config.rnn_dropout_state_keep_probability,
                    variational_recurrent=self.config.use_variational_dropout,
                    input_size=(input_layer.shape[-1]),
                    dtype=input_layer.dtype,
                )
                rnn_cell_bw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                    rnn_cell_bw,
                    input_keep_prob=self.config.rnn_dropout_input_keep_probability,
                    output_keep_prob=self.config.rnn_dropout_output_keep_probability,
                    state_keep_prob=self.config.rnn_dropout_state_keep_probability,
                    variational_recurrent=self.config.use_variational_dropout,
                    input_size=(input_layer.shape[-1]),
                    dtype=input_layer.dtype,
                )

                (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    rnn_cell_fw,
                    rnn_cell_bw,
                    input_layer,
                    sequence_length=self._input_sequence_length,
                    dtype=tf.float32
                )
                output = tf.concat([output_fw, output_bw], axis=-1)

                if self.config.short_cut_connections:
                    input_layer = tf.concat([output, self._embeddings_layer], axis=-1)
                else:
                    input_layer = output

            for task in output_here:
                self._shared_layers_output[task.name] = output

    def _build_task_termination(self):
        """
        Build task-specific nodes, losses, and optimizers.
        """
        logger = logging.getLogger("%s.Network._build_task_termination" % self.config.name)
        logger.debug("Building task termination")

        for task in self.config.tasks:
            input_layer = self._shared_layers_output[task.name]
            logger.debug("Building task termination for task %s on top of shared layers", task.name)
            logger.debug("Building %d hidden layers", len(task.hidden_layers))

            for idx, hidden_layer in enumerate(task.hidden_layers):
                assert isinstance(hidden_layer, HiddenLayerConfig)
                logger.debug(
                    "Building %d. hidden layer with %d units and activation %s",
                    idx + 1,
                    hidden_layer.units,
                    hidden_layer.activation
                )

                input_layer = tf.compat.v1.layers.dense(
                    input_layer,
                    hidden_layer.units,
                    activation=ACTIVATION_MAPPING[hidden_layer.activation],
                    name="hidden_layer-%s-%d" % (task.name, idx + 1)
                )

            input_layer = tf.nn.dropout(input_layer, 1 - (task.dropout_keep_probability))

            # Projection for prediction
            num_classes = len(task.data_reader.get_labels())
            logger.debug("Build projection layer to map network output to classes. There are %d classes", num_classes)

            self._projections[task.name] = tf.compat.v1.layers.dense(
                input_layer,
                num_classes,
                name="projection_layer-%s" % task.name
            )

            # Loss and prediction
            logger.debug("Attaching classifier")
            if task.classifier == CLASSIFIER_CRF:
                # CRF
                logger.debug("CRF classifier")
                # Prediction is performed via Viterbi decoding -> no prediction layer necessary
                self._predictions[task.name] = None
                with tf.compat.v1.variable_scope("crf_log_likelihood_%s" % task.name):
                    log_likelihood, self._transition_params[task.name] = tfa.text.crf_log_likelihood(
                        self._projections[task.name],
                        self._inputs_label[task.name],
                        self._input_sequence_length
                    )
                self._losses[task.name] = tf.reduce_mean(input_tensor=-log_likelihood)
            else:
                # Softmax
                logger.debug("Softmax classifier")
                self._predictions[task.name] = tf.cast(tf.argmax(input=self._projections[task.name], axis=-1), tf.int32)
                # Transition params are not required for softmax
                self._transition_params[task.name] = None

                labels = tf.one_hot(self._inputs_label[task.name], len(task.data_reader.get_labels()))

                # NOTE: this is for testing soft-label capability only (should be disabled!)
                # labels = tf.multiply(labels, 10.0)  # Multiply with 10 so that true label has a higher weight
                # labels = tf.add(labels, 1.0)  # Add one so that multiplication with random values has effect
                # noise = tf.random_uniform(
                #     tf.shape(labels)
                # )
                # labels = tf.multiply(labels, noise)  # Element-wise multiplication with noise
                # labels = tf.nn.softmax(labels)  # Perform softmax to restore the valid probability distribution

                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self._projections[task.name],
                    labels=tf.stop_gradient(labels),
                    name="softmax_%s" % task.name
                )
                # Add Mask for padded sentences
                mask = tf.sequence_mask(self._input_sequence_length, name="softmax_mask_%s" % task.name)
                losses = tf.boolean_mask(tensor=losses, mask=mask, name="softmax_mask_layer_%s" % task.name)
                self._losses[task.name] = tf.reduce_mean(input_tensor=losses)

            # Optimizer
            logger.debug("Attaching optimizer")
            optimizer_function = OPTIMIZER_MAPPING[self.config.training.optimizer]
            optimizer = optimizer_function(**self.config.training.optimizer_params)

            gradients, variables = list(zip(*optimizer.compute_gradients(self._losses[task.name])))

            if self.config.training.use_gradient_clipping:
                logger.debug("Adding node for performing gradient clipping for task %s.", task.name)
                gradients, self._gradient_norms[task.name] = tf.clip_by_global_norm(
                    gradients, self.config.training.clip_norm
                )
            else:
                self._gradient_norms[task.name] = tf.linalg.global_norm(gradients)

            self._operations_train[task.name] = optimizer.apply_gradients(list(zip(gradients, variables)))

    def build(self):
        """
        Build entire computation graph using the "private" methods, i.e. methods that start with an underscore.
        """
        start = time.time()
        logger = logging.getLogger("%s.Network.build" % self.config.name)
        logger.debug("Started building the computation graph")
        self._build_placeholders()
        self._build_embedding_layer()
        self._build_shared_layers()
        self._build_task_termination()
        self._build_initialization()
        logger.debug("Finished building the computation graph after %f seconds", time.time() - start)

    def _pre_process_input(self, task_name, batch, logger=None, verbose=False):
        """
        Pre-process the input given in the batch.
        This method creates the `feed_dict` required as an input for the Tensorflow network.

        Args:
            task_name (str): Name of the task for which the feed_dict shall be created (important for choosing the right
                label input node).
            batch (Batch): A batch object containing the batch's samples.
            logger (Logger, optional): A logger instance
            verbose (bool, optional): Whether to log something or not

        Returns:
            dict: `feed_dict` for the Tensorflow network
        """
        if logger is None:
            logger = logging.getLogger("%s.Network._pre_process_input" % self.config.name)

        # NOTE: it is assumed that all sentences within a batch are of the same length.
        #       See implementation of Batches for further information.

        # Special case for sentences of length 1
        sentences, labels = pad_sentences_of_length_one(batch.tokens, batch.labels, self.config.word2idx)

        if verbose:
            logger.debug(" - Batch size is %d", labels.shape[0])
            logger.debug(" - True sequence length is %d", batch.sequence_lengths[0])
            logger.debug(" - Sequence length is %d", labels.shape[1])
            logger.debug(" - Sentence shape %s", sentences.shape)
            logger.debug(" - Label shape %s", labels.shape)
            logger.debug(" - Sequence lengths shape %s", batch.sequence_lengths.shape)

        feed_dict = {
            self._input_word: sentences,
            self._inputs_label[task_name]: labels,
            self._input_sequence_length: batch.sequence_lengths
        }

        if self.config.character_level_information:
            if verbose:
                logger.debug(" - True character information shape is %s", batch.characters.shape)
                logger.debug(" - True word lengths shape is %s", batch.word_lengths.shape)
                logger.debug(" - Word length is %d", batch.word_lengths[0][0])

            characters, word_lengths = pad_character_sentences_of_length_one(
                characters=batch.characters,
                word_lengths_matrix=batch.word_lengths,
                char2idx=self.config.char2idx
            )

            if verbose:
                logger.debug(" - Character information shape is %s", characters.shape)
                logger.debug(" - Word lengths shape is %s", word_lengths.shape)

            feed_dict[self._input_characters] = characters
            feed_dict[self._input_word_length] = word_lengths

        return feed_dict

    @staticmethod
    def _get_tf_sess_config():
        """
        Builds a session configuration object for tensorflow.

        Returns:
            tf.ConfigProto: configuration object
        """
        sess_config = tf.compat.v1.ConfigProto()
        # Prevent tensorflow from reserving all GPU memory
        sess_config.gpu_options.allow_growth = True

        return sess_config

    def train(self, epochs=None, verbose=True, log_results_on_dev=True):
        """
        Train the network with data from the configuration.
        Args:
            epochs (int, optional): Number of epochs to train. This overwrites the settings from the configuration.
            verbose (bool, optional): Whether to print the progress. Defaults to True.
            log_results_on_dev (bool, optional): Whether to log the evaluation results on the development dataset
                after each epoch.

        Returns:
            (int, bool): A tuple with the number of actual epochs and a flag that indicates whether or not training
                stopped early.
        """
        train_start = time.time()

        logger = logging.getLogger("%s.Network.train" % self.config.name)
        logger.debug("Starting training")

        tasks = self.config.tasks
        logger.debug("Training %d tasks", len(tasks))
        for task in tasks:
            logger.debug(
                "Task %s: #words=%d; #labels=%d; classifier=%s; main=%s",
                task.name,
                len(task.data_reader.get_words()),
                len(task.data_reader.get_labels()),
                task.classifier,
                self.config.early_stopping is not None and self.config.early_stopping.task_name == task.name
            )

        # NOTE: chosen to be -1 so that the first epoch always saves the model
        best_score = -1
        saver = tf.compat.v1.train.Saver()
        model_out_path = os.path.join(self._paths["runs"][self._run_idx]["model"], PREFIX_MODEL_WEIGHTS)
        logger.debug("Best model weights will be stored in %s", self._paths["runs"][self._run_idx]["model"])
        logger.debug("The full path with prefix for model storage is %s", model_out_path)

        num_epoch_no_improvement = 0
        # Actual number of epochs in training (might be lower than configuration when using early stopping)
        num_actual_epochs = 0
        stopped_early = False

        batches = Batches(self.config, data_type=DATA_TYPE_TRAIN)
        logger.debug("Loaded batches of training data.")

        with tf.compat.v1.Session(config=self._get_tf_sess_config()) as sess:
            logger.debug("Initialize the network")
            sess.run(self._init)

            logger.debug("Logging the network graph")
            tf_writer = tf.compat.v1.summary.FileWriter(
                os.path.join(self._paths["runs"][self._run_idx]["out"], DIR_TENSOR_BOARD),
                sess.graph
            )

            epochs = epochs if epochs is not None else self.config.epochs
            for epoch in range(epochs):
                num_actual_epochs += 1
                epoch_start = time.time()
                logger.info("*" * 80)
                logger.info("Running epoch %d of %d epochs", epoch + 1, epochs)
                logger.info("*" * 80)

                num_batches = batches.find_total_num_batches()
                num_finished_batches = 0

                # Alternate training
                for task_name, batch in batches.iterate_batches_randomly():
                    assert isinstance(batch, Batch)
                    batch_start = time.time()
                    if verbose:
                        logger.debug("Running batch from task %s", task_name)

                    feed_dict = self._pre_process_input(task_name, batch, logger, verbose=verbose)

                    _, loss, norms = sess.run(
                        [self._operations_train[task_name], self._losses[task_name], self._gradient_norms[task_name]],
                        feed_dict=feed_dict
                    )

                    num_finished_batches += 1

                    logger.debug(
                        "Finished batch after %.4f seconds. Loss is %.4f. Gradient norm is: %.4f",
                        time.time() - batch_start,
                        loss,
                        norms
                    )
                    logger.debug("Finished %d of %d batches (%.2f%%)" % (
                        num_finished_batches,
                        num_batches,
                        (num_finished_batches / float(num_batches)) * 100
                    ))

                epoch_train_duration = time.time() - epoch_start
                # Only perform prediction on auxiliary task every 10 epochs
                result_lists = self.predict(sess, data_type=DATA_TYPE_DEV, only_main=epoch % 10 != 0)
                if log_results_on_dev:
                    logger.debug("Evaluating network on development data after epoch.")

                    for task_name, result_list in list(result_lists.items()):
                        assert isinstance(task_name, str)
                        assert isinstance(result_list, ResultList)
                        self.log_result_list(
                            task_name,
                            result_list,
                            "prediction_task-%s_epoch-%d" % (task_name, epoch)
                        )

                # Log to CSV
                epoch_duration = time.time() - epoch_start
                epoch_predict_duration = epoch_duration - epoch_train_duration
                for task_name, result_list in list(result_lists.items()):
                    assert isinstance(task_name, str)
                    assert isinstance(result_list, ResultList)

                    # Write a CSV file per task because each task may have different evaluation metrics
                    csv_out_path = os.path.join(
                        self._paths["runs"][self._run_idx]["out"],
                        "run_results.task_%s.csv" % task_name
                    )
                    self.log_result_list_csv(task_name, result_list, csv_out_path, {
                        "epoch": epoch + 1,
                        "epoch duration [sec]": epoch_duration
                    })

                if self.config.early_stopping is not None:
                    # Perform early stopping if necessary
                    main_task_results = result_lists[self.config.early_stopping.task_name]
                    assert isinstance(main_task_results, ResultList)
                    new_best_score = max(
                        best_score,
                        main_task_results.compute_metric_by_name(self.config.early_stopping.metric)
                    )

                    if new_best_score > best_score:
                        logger.info(
                            "New best score (%s) for the main task: %.3f",
                            self.config.early_stopping.metric.title(),
                            new_best_score
                        )
                        best_score = new_best_score
                        num_epoch_no_improvement = 0
                        saver.save(sess, model_out_path)
                    else:
                        num_epoch_no_improvement += 1
                        logger.debug("No improvement for main task in this epoch. "
                                     "Best is %.3f. "
                                     "This is the %d. epoch without improvement. "
                                     "Stopping after %d epochs without improvement.",
                                     best_score,
                                     num_epoch_no_improvement,
                                     self.config.early_stopping.patience)

                        if num_epoch_no_improvement >= self.config.early_stopping.patience:
                            logger.info(
                                "Early stopping because there was no improvement for %d epochs.",
                                self.config.early_stopping.patience
                            )
                            stopped_early = True
                            logger.debug("Finished epoch after %.4f seconds", epoch_duration)
                            break
                else:
                    saver.save(sess, model_out_path)

                logger.debug("Finished epoch after %.4f seconds", epoch_duration)
                if verbose:
                    logger.debug(
                        "Epoch consists of %.4f seconds (%.3f%%) train time and %.4f seconds (%.3f%%) prediction time",
                        epoch_train_duration,
                        (epoch_train_duration / epoch_duration) * 100,
                        epoch_predict_duration,
                        (epoch_predict_duration / epoch_duration) * 100,
                    )

            tf_writer.close()

        duration = time.time() - train_start
        logger.debug("Finished training after %.4f seconds", duration)
        self.log_duration_csv(duration, "train", num_actual_epochs, stopped_early)
        return num_actual_epochs, stopped_early

    def predict(self, sess, data_type=DATA_TYPE_DEV, only_main=False):
        """
        Perform prediction for data of the specified type and return the prediction results together with metrics.
        Args:
            sess (object): Tensorflow session
            data_type (str): Which type of data to use for prediction (usually dev or test)
        Returns:
            `dict` of ResultList: a result list for each task
        """
        predict_start = time.time()
        logger = logging.getLogger("%s.Network.predict" % self.config.name)
        logger.debug("Starting prediction on %s data set", data_type)

        batches_pkl = os.path.join(self._paths["runs"][self._run_idx]["batches"], "prediction_%s.pkl" % data_type)

        if os.path.isfile(batches_pkl):
            logger.debug("Loading %s batches from pickle file.", data_type)
            with open(batches_pkl, "rb") as f:
                batches = pkl.load(f)
        else:
            logger.debug("No pickle file for %s batches. Creating them from scratch.", data_type)
            batches = Batches(self.config, data_type=data_type, no_mini_batches=True)
            logger.debug("Storing batches for %s in pickle file.", data_type)
            with open(batches_pkl, "wb") as f:
                pkl.dump(batches, f, -1)
        logger.debug("Finished loading batches")

        prediction_results = {
            task.name: []
            for task in self.config.tasks
        }
        result_lists = {}

        uses_crf = {
            task.name: task.classifier == CLASSIFIER_CRF
            for task in self.config.tasks
        }

        for task_name, batch in batches.iterate_tasks():
            if only_main and self.config.early_stopping is not None and task_name != self.config.early_stopping.task_name:
                # logger.debug(
                #     "Skipping prediction for task %s because only main task (%s) is predicted.",
                #     task_name,
                #     self.config.early_stopping.task_name
                # )
                continue

            assert isinstance(batch, Batch)
            samples = batch.samples
            # NOTE: it is assumed that all sentences within a batch are of the same length.
            #       See implementation of Batches for further information.

            # Special case for sentences of length 1
            sentences, labels = pad_sentences_of_length_one(batch.tokens, batch.labels, self.config.word2idx)

            feed_dict = self._pre_process_input(task_name, batch, logger)

            if uses_crf[task_name]:
                projections, transition_params = sess.run(
                    [self._projections[task_name], self._transition_params[task_name]],
                    feed_dict=feed_dict
                )

                predictions = []
                for projection in projections:
                    viterbi_sequence, _ = tfa.text.viterbi_decode(
                        projection,
                        transition_params
                    )
                    predictions += [viterbi_sequence]

                prediction_results[task_name].append((sentences, labels, predictions, samples))
            else:
                predictions = sess.run([self._predictions[task_name]], feed_dict=feed_dict)
                prediction_results[task_name].append((sentences, labels, predictions[0], samples))

        for task_name, result_list in list(prediction_results.items()):
            if len(result_list) == 0:
                # Ignore empty result lists
                continue

            logger.debug("Checking predictions for task %s", task_name)

            # Flatten the batches into a single list.
            # This list consists of tuples of sentence, labels, predicted labels, and sample object.
            # NOTE: there is no padding at this point!
            flattened_results = []
            for sentences, labels, predictions, samples in result_list:
                flattened_results.extend(list(zip(sentences, labels, predictions, samples)))

            task = self._task_by_name[task_name]
            result_lists[task_name] = ResultList(
                flattened_results,
                task.data_reader.get_labels(out_format="label2idx"),
                task
            )

        duration = time.time() - predict_start
        logger.debug("Finished prediction after %.4f seconds", duration)
        self.log_duration_csv(duration, "predict")
        return result_lists

    def evaluate(self, data_type=DATA_TYPE_DEV, model_path=None):
        """
        Evaluate the network with the specified data set.

        Args:
            data_type (str): Which type of data set to use for evaluation
            model_path (str): Path to a stored model

        Returns:
            `dict` of ResultList: Results for the evaluation
        """
        logger = logging.getLogger("%s.Network.evaluate" % self.config.name)
        logger.debug("Evaluating %s data.", data_type)

        with tf.compat.v1.Session(config=self._get_tf_sess_config()) as sess:
            saver = tf.compat.v1.train.Saver()
            model_path = model_path if model_path is not None else self._paths["runs"][self._run_idx]["model"]
            model_out_path = os.path.join(model_path, PREFIX_MODEL_WEIGHTS)
            logger.debug("Restoring session from %s.", model_out_path)
            saver.restore(sess, model_out_path)

            result_lists = self.predict(sess, data_type=data_type)

            for task_name, result_list in list(result_lists.items()):
                assert isinstance(task_name, str)
                assert isinstance(result_list, ResultList)
                self.log_result_list(
                    task_name,
                    result_list,
                    "evaluation_data-%s_prediction_task-%s" % (data_type, task_name)
                )

        return result_lists

    def log_result_list(self, task_name, result_list, prediction_out_file_name):
        """
        Log the result list and write predictions to files.

        Args:
            task_name (str): The task's name
            result_list (ResultList): Prediction results for a single data file of a task (usually dev or test)
            prediction_out_file_name (str): The name (not the path!) of the file that holds the predictions
        """
        assert isinstance(task_name, str)
        assert isinstance(result_list, ResultList)
        assert isinstance(prediction_out_file_name, str)

        logger = logging.getLogger("%s.Network.log_result_list" % self.config.name)

        task = self._task_by_name[task_name]

        assert isinstance(task, TaskConfig)

        logger.debug("Logging results for metrics %s and %s", self.config.eval_metrics, task.eval_metrics)

        # Merge global and task-specific metrics
        for metric in set(self.config.eval_metrics + task.eval_metrics):
            logger.info(
                "%s at task %s is %.3f",
                metric.title(),
                task_name,
                result_list.compute_metric_by_name(metric)
            )

        result_list.predictions_to_file(self._paths["runs"][self._run_idx]["predictions"], prediction_out_file_name)

    def log_duration_csv(self, duration, task, num_epochs=0, stopped_early=False):
        """
        Write the duration for the task (train or predict) to a CSV file.

        Args:
            duration (float): Duration in seconds
            task (str): Task; either "train" or "predict"
            num_epochs (int): Number of training epochs
            stopped_early (bool): Whether the training was stopped early
        """
        assert isinstance(duration, float)
        assert task in ["train", "predict"]

        csv_file_name = "duration.csv"
        csv_file_path = os.path.join(self._paths["session_out"], csv_file_name)

        headers = ["timestamp", "task", "duration [sec]", "epochs", "stopped early"]
        values = [time.strftime("%Y-%m-%d %H:%M"), task, duration, num_epochs, stopped_early]

        append_to_csv(csv_file_path, headers=headers, values=values)

    def log_result_list_csv(self, task_name, result_list, csv_file_path, additional_values=None):
        """
        Append results to a CSV file.

        Args:
            task_name (str): Name of the task that has been evaluated
            result_list (ResultList): Evaluation results
            csv_file_path (str): Path to CSV file
            additional_values (dict): A dictionary of additional values that shall be added to the CSV file. The keys
                of the dictionary are used as headers.
        """
        if additional_values is None:
            additional_values = {}
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        headers = ["timestamp", "session_id", "task_name"]
        values = [timestamp, self._session_id, task_name]

        for k, v in list(additional_values.items()):
            headers.append(k)
            values.append(v)

        task = self._task_by_name[task_name]
        assert isinstance(task, TaskConfig)

        # Merge global and task-specific metrics
        for metric in set(self.config.eval_metrics + task.eval_metrics):
            headers.append(metric.title())
            values.append(result_list.compute_metric_by_name(metric))

        append_to_csv(csv_file_path, headers=headers, values=values)


# TODO: replace this by a proper solution
def pad_sentences_of_length_one(sentences, labels, word_2_idx):
    """
    If the provided sentences have length one, they are padded.

    Args:
        sentences (np.ndarray): Matrix with shape (batch size, sequence length) containing token indices.
        labels (np.ndarray): Matrix with shape (batch size, sequence length) containing label indices.
        word_2_idx (`dict` of int): Mapping from words to indices.
    Returns:
        `tuple` of np.ndarray: Sentence and label matrix (in this order)
    """
    if sentences.shape[1] != 1:
        # Do not change if the length is not one
        return sentences, labels

    return np.asarray([
        np.concatenate((sentence, np.asarray([word_2_idx[TOKEN_PADDING]])))
        for sentence in sentences
    ]), np.asarray([
        np.concatenate((label_sequence, np.asarray([0])))
        for label_sequence in labels
    ])


def pad_character_sentences_of_length_one(characters, word_lengths_matrix, char2idx):
    """
    If the provided sentences (represented by lists of character indices) have length one, they are padded.

    Args:
        characters (np.ndarray): Tensor with shape (batch size, sequence length, word length)
        word_lengths_matrix (np.ndarray): Matrix with shape (batch size, sequence length)
        char2idx (`dict` of int): Mapping from characters to indices.

    Returns:
        `tuple` of np.ndarray: Character tensor and word length matrix
    """
    assert len(characters.shape) == 3
    assert len(word_lengths_matrix.shape) == 2
    assert isinstance(char2idx, dict)

    batch_size, sequence_length, word_length = characters.shape

    if sequence_length != 1:
        # Do not change if the length is not one
        return characters, word_lengths_matrix

    character_padding = np.ones((1, word_length)) * char2idx[TOKEN_PADDING]
    word_length_scalar = word_lengths_matrix[0][0]
    word_length_padding = np.ones((1,)) * word_length_scalar

    padded_characters = []
    padded_word_lengths = []

    for i in range(batch_size):
        sentence = characters[i]
        assert sentence.shape[0] == sequence_length
        assert sentence.shape[1] == word_length

        length_vector = word_lengths_matrix[i]
        assert length_vector.shape[0] == sequence_length

        padded_sentence = np.concatenate((sentence, character_padding))
        assert padded_sentence.shape[0] == sequence_length + 1
        padded_length_vector = np.concatenate((length_vector, word_length_padding))
        assert padded_length_vector.shape[0] == sequence_length + 1

        padded_characters.append(padded_sentence)
        padded_word_lengths.append(padded_length_vector)

    return np.asarray(padded_characters), np.asarray(padded_word_lengths)
