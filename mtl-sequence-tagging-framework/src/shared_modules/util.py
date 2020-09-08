"""Utilities

This module contains utility functions that are used frequently.
"""
import csv
import json
import logging
import logging.config
import numpy as np
import os
import uuid
from distutils.dir_util import mkpath
from shutil import copy

import time
from keras.utils import np_utils

from .config.ExperimentConfig import ExperimentConfig
from .constants import ALIGNMENT_STRATEGY_RANDOM_SAMPLE, ALIGNMENT_STRATEGY_CROP, DIR_MODEL_WEIGHTS, DIR_PREDICTION_OUT, DIR_TENSOR_BOARD, \
    DIR_RUN, DIR_BATCHES_OUT
from .constants import DIR_OUT, DIR_SRC, DIR_DATA
from .constants import DIR_PKL


def swap_dict(input_dict):
    """
    Swap keys and values of the provided dictionary so that the returned
    dictionary has the values as keys and the keys as values.

    Args:
        input_dict (dict): Input dictionary

    Returns:
        dict: swapped dictionary
    """
    return {k: v for v, k in input_dict.items()}


def separate_list_of_tuples(list_of_tuples):
    """
    From a list of tuples each having n elements, create n lists of elements where the first
    list only contains all the first items in the elements, the second list all the second items, etc.
    Args:
        list_of_tuples (`list` of `tuple` of object): A list of tuples.

    Returns:
        (`tuple` of `list` of object): A tuple of lists.
    """
    if not list_of_tuples or len(list_of_tuples) == 0:
        return ()

    num_elements = len(list_of_tuples[0])

    return ([tup[i] for tup in list_of_tuples] for i in xrange(num_elements))


def list_of_lists_to_list_of_tensors(list_of_lists):
    """
    Convert a list of lists into a list of tensors (`np.ndarray`).
    Args:
        list_of_lists (`list` of `list` of object): A list of lists.

    Returns:
        (`list` of np.ndarray): A list of tensors.
    """
    return [np.asarray(lst) for lst in list_of_lists]


def predictions_to_classes(predictions, idx_2_label):
    """
    Convert a list of predictions (list of dense vectors with real values) to
    a list of classes. Of course, also works for one-hot vectors (usually from
    gold data) instead of predictions.
    Args:
        predictions (`list` of `list` of float or np.ndarray): Predictions.
        idx_2_label (`dict` of str): A mapping from label indices to label names.

    Returns:
        `list` of `list` of str: A list of label sequences.
    """
    classes = []

    for vector in predictions:
        class_idx = np.argmax(vector)
        # noinspection PyTypeChecker
        classes.append(idx_2_label[class_idx])

    return classes


def idx_to_one_hot(data, num_classes):
    """
    Converts a lists of indexes (can also be padded) into a numpy array of one-hot vectors.

    Args:
        data (`list` of np.ndarray or np.ndarray): A list of index lists.
        num_classes (int): The number of classes

    Returns:
        np.ndarray: A list of one-hot vector lists.
    """
    return np.asarray(
        [
            np_utils.to_categorical(idx_list, num_classes=num_classes)
            for idx_list in data
        ],
        dtype="int32"
    )


def align_list_length(lists, mode=ALIGNMENT_STRATEGY_RANDOM_SAMPLE):
    """
    Ensure that the length of all provided lists is the same after calling this method.
    The provided mode allows to decide how the length alignment is performed:

        * `ALIGNMENT_STRATEGY_RANDOM_SAMPLE`: lists that are shorter than the longest list
            are extended by adding duplicate entries that are randomly sampled from the list.
        * `ALIGNMENT_STRATEGY_CROP`: all lists are cropped to have the same length, i.e. the
            minimum length.

    Args:
        lists (`list` of `list` of object or `tuple` of `list` of object): A list/tuple of lists.
        mode (str): the alignment strategy. Either `ALIGNMENT_STRATEGY_RANDOM_SAMPLE` or
            `ALIGNMENT_STRATEGY_CROP`.

    Returns:
        A list of lists that all have the same length.
    """
    assert isinstance(lists, list) or isinstance(lists, tuple)
    assert mode in [ ALIGNMENT_STRATEGY_RANDOM_SAMPLE, ALIGNMENT_STRATEGY_CROP ]

    # No action necessary for only one list
    if len(lists) == 1:
        return lists

    all_lengths = [len(lst) for lst in lists]
    max_length = max(all_lengths)
    min_length = min(all_lengths)

    # No action necessary because all lists already have the same length
    if all([length == max_length for length in all_lengths]):
        return lists

    aligned_lists = []

    if mode == ALIGNMENT_STRATEGY_RANDOM_SAMPLE:
        for i, lst in enumerate(lists):
            lst_length = len(lst)
            if lst_length < max_length:
                num_missing_entries = max_length - lst_length
                new_entries = [lst[j] for j in np.random.choice(lst_length, num_missing_entries)]
                # Handle native python lists as well as numpy arrays
                # Numpy arrays are concatenated along the first axis
                new_lst = lst + new_entries if isinstance(lst, list) else np.concatenate(
                    (lst, new_entries)
                )
                aligned_lists.append(new_lst)
            else:
                aligned_lists.append(lst)
    elif mode == ALIGNMENT_STRATEGY_CROP:
        aligned_lists = [lst[:min_length] for lst in lists]
    else:
        raise ValueError("Unknown alignment mode '%s' for `align_list_length`" % mode)

    return aligned_lists


def setup(path_to_config, num_runs=None, log_level=logging.DEBUG):
    """
    Setup an experiment by determining all necessary paths (out, pkl, src, data)
    and reading and preparing the configuration.

    Args:
        path_to_config (str): Path to the configuration file. This is also used to determine the experiment's root
            folder.
        num_runs (int, optional): Number of runs for the experiment. When specifiec, overrides the setting in the
            configuration file.
        log_level (int, optional): Log level for the logger.

    Returns:
        `tuple` of object: A tuple consisting of the configuration object, a path dict with all necessary paths, and the
            session id.
    """
    assert isinstance(path_to_config, str)
    assert num_runs is None or (isinstance(num_runs, int) and num_runs > 0)
    assert os.path.exists(path_to_config)
    assert os.path.isfile(path_to_config)

    abs_path_to_config = os.path.abspath(path_to_config)
    # Find root path by determining which directory has "experiments" as its parent
    abs_path_components = abs_path_to_config.split(os.path.sep)

    if "experiments" in abs_path_components:
        parent = os.path.sep.join(abs_path_components[:abs_path_components.index("experiments") + 2])
    elif "mtl-sequence-tagging-framework" in abs_path_components:
        # Try to find the parent by a specific foler name
        parent = os.path.sep.join(abs_path_components[:abs_path_components.index("mtl-sequence-tagging-framework") + 1])
    else:
        assert False, "Could not find experiment directory. It should either be located within the directory " \
                      "'experiments' or its name should be 'mtl-sequence-tagging-framework'"

    config = ExperimentConfig(abs_path_to_config)
    config.read()

    assert config.name != "", "Expected configuration to have a name"

    session_id = time.strftime("%Y-%m-%d_%H%M") + "_" + uuid.uuid4().hex

    # Set num_runs if it was not supplied as a parameter
    if num_runs is None:
        num_runs = config.num_runs

    # Output paths
    out_path = os.path.join(parent, DIR_OUT)
    experiment_out_path = os.path.join(out_path, config.name)
    session_out_path = os.path.join(experiment_out_path, session_id)
    run_out_paths = {idx: {
        "out": os.path.join(session_out_path, DIR_RUN % (idx + 1)),
        "predictions": os.path.join(session_out_path, DIR_RUN % (idx + 1), DIR_PREDICTION_OUT),
        "model": os.path.join(session_out_path, DIR_RUN % (idx + 1), DIR_MODEL_WEIGHTS),
        "batches": os.path.join(session_out_path, DIR_RUN % (idx + 1), DIR_BATCHES_OUT),
    } for idx in range(num_runs)}

    pkl_path = os.path.join(parent, DIR_PKL, config.name)
    src_path = os.path.join(parent, DIR_SRC)
    data_path = os.path.join(parent, DIR_DATA)

    paths = {
        "out": out_path,
        "experiment_out": experiment_out_path,
        "session_out": session_out_path,
        "pkl": pkl_path,
        "src": src_path,
        "data": data_path,
        "runs": run_out_paths
    }

    # Create experiment-specific folders
    mkpath(out_path)
    mkpath(pkl_path)
    for path_dict in run_out_paths.values():
        for path in path_dict.values():
            mkpath(path)

    # Setup logging
    setup_logging(config.name, session_out_path, log_level)
    logger = logging.getLogger(config.name)
    logger.info("Setting up experiment %s", config.name)
    logger.info("Session ID: %s", session_id)
    logger.info("Running the experiment %d times.", num_runs)

    # Check validity of config
    logger.debug("Checking validity of configuration")
    config_valid = config.sanity_check()

    if not config_valid:
        logger.error("Config is not valid.")
        raise Exception("Config is not valid.")

    logger.debug("Config is valid.")

    # Set paths
    logger.debug("Setting paths for config. Paths: %s", paths)
    config.set_paths(paths)

    # Prepare config
    logger.debug("Preparing configuration")
    config_prepared = config.prepare()

    if not config_prepared:
        logger.error("Could not prepare configuration for further use.")
        raise Exception("Could not prepare configuration for further use.")

    json_config = json.dumps(config.to_dict(), sort_keys=True, indent=4)

    logger.debug("Parsed and prepared configuration: %s", json_config)
    logger.debug("Configured paths: %s", json.dumps(paths, sort_keys=True, indent=4))

    config_store_path = os.path.join(session_out_path, "config.json")
    logger.debug("Storing session configuration on disk at %s.", config_store_path)
    with open(config_store_path, mode="w") as f:
        f.write(json_config)

    # Make a copy of the original YAML file as well
    copy(path_to_config, session_out_path)

    return config, paths, session_id


def setup_logging(name, log_path="../log", level=logging.INFO):
    """
    Setup logging for an experiment.
    Args:
        name (str): Name of the experiment.
        log_path (str): Where to store the log files.
        level (int): Log level.

    Returns:
        Nothing.
    """
    config = {
        "version": 1,
        "handlers": {
            "fileHandler": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": "%s/%s.log" % (log_path, name)
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            name: {
                "handlers": ["fileHandler", "console"],
                "level": level
            },
            "shared": {
                "handlers": ["fileHandler", "console"],
                "level": level
            },
            "root": {
                "handlers": ["fileHandler", "console"],
                "level": level
            }
        },
        "formatters": {
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    }
    logging.config.dictConfig(config)
    logger = logging.getLogger("shared")
    logger.debug("Finished setting up logging for app '%s' and log path '%s'", name, log_path)


def append_to_csv(file_path, headers, values):
    """
    Append the values to the CSV file at `file_path`.
    If the file does not exist, it is created with the provided headers.

    Args:
        file_path (str): Path to CSV file
        headers (`list` of str): List of headers
        values(`list` of str): List of values

    Returns:

    """
    assert isinstance(headers, list)
    assert isinstance(values, list)
    assert len(headers) == len(values)

    logger = logging.getLogger("shared.write_to_csv")
    logger.debug("Appending to CSV file at %s", file_path)

    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a") as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_ALL)

        if not file_exists:
            logger.debug("CSV file does not exist yet. Creating a new one with the provided headers.")
            logger.debug("Headers: %s", headers)
            csv_writer.writerow(headers)

        csv_writer.writerow(values)
