"""Class to create batches from a configuration"""
import logging
import math
import random

import numpy as np

from .Batch import Batch
from ..config.ExperimentConfig import ExperimentConfig
from ..constants import DATA_OUT_INDEX, DATA_TYPE_TEST, DATA_TYPE_DEV
from ..constants import DATA_TYPE_TRAIN


class Batches(object):
    def __init__(self, config, data_type=DATA_TYPE_TRAIN, no_mini_batches=False):
        """
        Initialize the batches by loading the training data from the configuration file.

        Populates the `_batches` property with a dict that matches each task to batches.
        Each item in a batch is a tuple of list of labels, list of tokens, and list of sample objects.

        Args:
            config (ExperimentConfig): Configuration object
            data_type (str): type of data (train, dev or test)
            no_mini_batches (bool): whether to use mini batches or not, i.e. whether to use the batch_size specified
                in the configuration file or just build one batch for each sequence length
        """
        assert isinstance(config, ExperimentConfig)
        assert data_type in [DATA_TYPE_TRAIN, DATA_TYPE_DEV, DATA_TYPE_TEST]
        logger = logging.getLogger("shared.Batches.__init__")

        self._batches = {}

        logger.debug("Building batches for %d tasks", len(config.tasks))
        for task in config.tasks:
            logger.debug("Task: %s", task.name)
            data = task.data_reader.get_data(data_type, DATA_OUT_INDEX, word2idx=config.word2idx)
            # Sort by sentence length
            data.sort(key=lambda sample: sample.len)
            train_ranges = []
            old_sent_length = data[0].len
            idx_start = 0

            # Find start and end of ranges with sentences with same length
            for idx in range(len(data)):
                sent_length = data[idx].len

                if sent_length != old_sent_length:
                    train_ranges.append((idx_start, idx))
                    idx_start = idx

                old_sent_length = sent_length

            # Add last sentence
            train_ranges.append((idx_start, len(data)))
            logger.debug("%d different sentence lengths", len(train_ranges))

            # Break up ranges into smaller mini batch sizes
            mini_batch_ranges = []
            if no_mini_batches:
                logger.debug("`no_mini_batches == True` --> not building any mini batches")
            for batch_range in train_ranges:
                range_len = batch_range[1] - batch_range[0]

                if no_mini_batches:
                    bins = 1
                else:
                    bins = int(math.ceil(range_len / float(config.batch_size)))

                bin_size = int(math.ceil(range_len / float(bins)))

                for binNr in range(bins):
                    start_idx = binNr * bin_size + batch_range[0]
                    end_idx = min(batch_range[1], (binNr + 1) * bin_size + batch_range[0])
                    mini_batch_ranges.append((start_idx, end_idx))

            logger.debug("%d batches", len(mini_batch_ranges))

            self._batches[task.name] = []

            # Shuffle training data
            # 1. Shuffle sentences that have the same length
            for data_range in train_ranges:
                for i in reversed(range(data_range[0] + 1, data_range[1])):
                    # pick an element in x[:i+1] with which to exchange x[i]
                    j = random.randint(data_range[0], i)
                    data[i], data[j] = data[j], data[i]

            # 2. Shuffle the order of the mini batch ranges
            random.shuffle(mini_batch_ranges)

            for rng in mini_batch_ranges:
                start, end = rng
                rng_samples = data[start:end]
                labels = np.asarray([sample.labels_as_array for sample in rng_samples])
                tokens = np.asarray([sample.tokens_as_array for sample in rng_samples])

                characters = None
                if config.character_level_information:
                    max_token_length = max([sample.get_max_token_length() for sample in rng_samples])
                    characters = np.asarray([
                        sample.get_tokens_as_char_ids(config.char2idx, max_token_length)
                        for sample in rng_samples
                    ])

                self._batches[task.name].append(Batch(labels, tokens, rng_samples, characters))

    def iterate_tasks(self):
        """
        Iterate over all batches by task, i.e. first use all batches for one task, then for the next and so on.
        Returns:
            generator: A generator for batches. The generator always returns a tuple consisting of the task name and
                the batch.
        """
        return ((task, batch) for task, batches in self._batches.items() for batch in batches)

    def find_min_num_batches(self):
        """
        Find the minimum number of batches across all tasks.
        This number is the number of batches for the iterate_batches method.

        Returns:
            int: minimum number of batches
        """
        logger = logging.getLogger("shared.Batches.find_min_num_batches")

        # min([len(batches) for batches in self._batches.values()])
        min_num_batches = float("inf")

        for task_name, batches in self._batches.items():
            num_batches = len(batches)
            if num_batches < min_num_batches:
                min_num_batches = num_batches

            logger.debug("Task %s has %d batches.", task_name, num_batches)

        logger.debug("Choosing the minimum for this iteration. Minimum: %d", min_num_batches)

        return min_num_batches

    def iterate_batches(self):
        """
        Iterate over all batches and alternate between tasks, i.e. use a batch of one task, then one of the next, and so
        on.
        Returns:
            generator: A generator for batches. The generator always returns a tuple consisting of the task name and
                the batch.
        """
        min_num_batches = self.find_min_num_batches()

        for idx in range(min_num_batches):
            for task in self._batches.keys():
                yield task, self._batches[task][idx]

    def find_total_num_batches(self):
        """
        Find the total number of batches across all tasks.

        Returns:
            int: total number of batches
        """
        return sum([len(batches) for batches in self._batches.values()])

    def iterate_batches_randomly(self):
        """
        Iterate over all batches and choose a batch from a task at random each time.

        Returns:
            generator: A generator for batches. The generator always returns a tuple consisting of the task name and
                the batch.
        """
        logger = logging.getLogger("shared.Batches.iterate_batches_randomly")

        tasks = list(self._batches.keys())
        num_tasks = len(tasks)

        task_indices = {task: 0 for task in tasks}
        task_num_batches = {task: len(self._batches[task]) for task in tasks}

        num_batches_total = self.find_total_num_batches()

        logger.debug("Iterating randomly over batches for %d tasks.", num_tasks)
        logger.debug("There are %d batches in total.", num_batches_total)
        for task, num_batches in task_num_batches.items():
            logger.debug("Task %s has %d batches.", task, num_batches)

        def find_task_with_remaining_batches():
            task = tasks[np.random.randint(0, num_tasks)]
            t_idx = task_indices[task]
            t_num_batches = task_num_batches[task]

            if t_idx < t_num_batches:
                return task
            else:
                return find_task_with_remaining_batches()

        for idx in range(num_batches_total):
            current_task = find_task_with_remaining_batches()
            logger.debug(
                "Current task %s; task index: %d; task batches: %d",
                current_task,
                task_indices[current_task],
                task_num_batches[current_task],
            )
            yield current_task, self._batches[current_task][task_indices[current_task]]
            task_indices[current_task] += 1


