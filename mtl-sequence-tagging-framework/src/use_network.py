"""
This module provides methods to use the network for training, prediction, and evaluation.
The provided methods take care of loading the required configuration files.
"""

import logging
import os
import time

import tensorflow as tf

from Network import Network
from shared_modules.config.ExperimentConfig import ExperimentConfig
from shared_modules.constants import DATA_TYPE_TEST, DATA_TYPE_DEV
from shared_modules.eval.ResultList import ResultList
from shared_modules.util import setup, append_to_csv


def train(path_to_config, verbose=False):
    """
    Train the network for multiple runs (can be specified in the configuration file with the "num_runs" option).
    After finishing the training for a run, the network is evaluated on the development data and the result is stored.
    After finishing all runs, the results are averaged.

    Args:
        path_to_config (str): Path to the configuration file.
        verbose (bool): Whether or not to display additional logging information
    """
    config, paths, session_id = setup(path_to_config)
    assert isinstance(config, ExperimentConfig)
    logger = logging.getLogger("%s.train" % config.name)

    results = []

    # Average results over `config.num_runs` runs
    for i in range(config.num_runs):
        logger.info("*" * 80)
        logger.info("* %d. run for experiment %s", (i + 1), config.name)
        logger.info("*" * 80)
        network = Network(config, paths, session_id, i)

        network.build()
        num_actual_epochs, stopped_early = network.train(verbose=verbose, log_results_on_dev=True)
        run_results = network.evaluate(data_type=DATA_TYPE_DEV)
        results.append(run_results)

        for task_name, result_list in list(run_results.items()):
            assert isinstance(task_name, str)
            assert isinstance(result_list, ResultList)
            # Write a CSV file per task because each task may have different evaluation metrics
            csv_out_path = os.path.join(paths["session_out"], "session_results.task_%s.csv" % task_name)
            network.log_result_list_csv(task_name, result_list, csv_out_path, {
                "# planned epochs": config.epochs,
                "# actual epochs": num_actual_epochs,
                "stopped early?": stopped_early,
                "run": i + 1
            })

        logger.info("*" * 80)
        logger.info("")

        # Reset tensorflow variables
        tf.compat.v1.reset_default_graph()

    logger.info("")
    logger.info("Results after %d runs:", config.num_runs)

    timestamp = time.strftime("%Y-%m-%d %H:%M")

    for task in config.tasks:
        task_name = task.name
        task_results = [result[task_name] for result in results]
        # Write a CSV file per task because each task may have different evaluation metrics
        csv_file_path = os.path.join(paths["experiment_out"], "results.task_%s.csv" % task_name)
        logger.info(" - Task %s", task_name)

        headers = ["timestamp", "session_id", "num_runs", "task_name"]
        values = [timestamp, session_id, config.num_runs, task_name]

        for metric in set(config.eval_metrics + task.eval_metrics):
            metric_values_sum = 0

            for result in task_results:
                metric_values_sum += result.compute_metric_by_name(metric)

            logger.info(
                "  - Average %s at task %s is %.3f",
                metric.title(),
                task_name,
                metric_values_sum / float(config.num_runs)
            )

            headers += ["AVG:%s" % metric.title()]
            values += [metric_values_sum / float(config.num_runs)]

        append_to_csv(csv_file_path, headers=headers, values=values)


def evaluate(path_to_config, path_to_model):
    """
    Evaluate the network on test data using the model stored in `path_to_model`.

    Args:
        path_to_config (str): Path to configuration file
        path_to_model (str): Path to the saved model
    """

    config, paths, session_id = setup(path_to_config, 1)
    assert isinstance(config, ExperimentConfig)
    logger = logging.getLogger("%s.main" % config.name)

    logger.info("Evaluating network on test data")

    network = Network(config, paths, session_id, 0)
    network.build()
    network.evaluate(DATA_TYPE_TEST, model_path=path_to_model)


def predict(path_to_config, path_to_model):
    # TODO: add special handling for files without labels
    raise NotImplementedError()
