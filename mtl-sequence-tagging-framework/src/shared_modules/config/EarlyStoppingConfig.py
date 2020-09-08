"""Class for configuring early stopping"""

from .BaseConfig import BaseConfig
from ..constants import METRIC_F1, VALID_METRICS


class EarlyStoppingConfig(BaseConfig):
    def __init__(self, task_name, metric=METRIC_F1, patience=5):
        """Initialize the file configuration.

        Args:
            task_name (str): main task for early stopping
            metric (str): which metric is used for early stopping
            patience (int): how many epochs to wait before early stopping
        """
        # Ensure that data types are correct
        assert isinstance(task_name, str)
        assert metric in VALID_METRICS
        assert isinstance(patience, int) and patience >= 0

        self._task_name = task_name
        self._metric = metric
        self._patience = patience

        self._prepared = False
        self._paths = {}
        self._paths_set = False

    @property
    def task_name(self):
        """str: main task for early stopping"""
        return self._task_name

    @property
    def metric(self):
        """str: which metric is used for early stopping"""
        return self._metric

    @property
    def patience(self):
        """int: how many epochs to wait before early stopping"""
        return self._patience

    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        self._prepared = True
        return True

    def sanity_check(self):
        return True

    def to_dict(self):
        return {
            "task_name": self.task_name,
            "metric": self.metric,
            "patience": self.patience,
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
        self._paths_set = True

    @property
    def paths_set(self):
        """
        Check if the paths have been set. Should be true after `set_paths` has been called.
        Returns:
            bool: True if the paths have been set, False otherwise.
        """
        return self._paths_set
