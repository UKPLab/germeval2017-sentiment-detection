"""Class for configuring training"""

from .BaseConfig import BaseConfig
from ..constants import VALID_OPTIMIZERS


class TrainingConfig(BaseConfig):
    def __init__(self, optimizer, optimizer_params, use_gradient_clipping, clip_norm):
        """Initialize the training configuration.

        Args:
            optimizer (str): name of the optimizer to use
            optimizer_params (object): parameters for the optimizer (see Tensorflow documentation)
            use_gradient_clipping (bool): whether or not to use gradient clipping
            clip_norm (float): clip norm for gradient clipping. Ignored if `use_gradient_clipping` is False.
        """
        # Ensure that data types are correct
        assert isinstance(optimizer, str)
        assert isinstance(optimizer_params, object)
        assert isinstance(use_gradient_clipping, bool)
        assert isinstance(clip_norm, float)

        self._optimizer = optimizer
        self._optimizer_params = optimizer_params
        self._use_gradient_clipping = use_gradient_clipping
        self._clip_norm = clip_norm

        self._prepared = False
        self._paths = {}
        self._paths_set = False

    @property
    def optimizer(self):
        """

        Returns:
            (str): name of the optimizer to use
        """
        return self._optimizer

    @property
    def optimizer_params(self):
        """

        Returns:
            (object): parameters for the optimizer (see Tensorflow documentation)
        """
        return self._optimizer_params

    @property
    def use_gradient_clipping(self):
        """

        Returns:
            (bool): whether or not to use gradient clipping
        """
        return self._use_gradient_clipping

    @property
    def clip_norm(self):
        """

        Returns:
            (float): clip norm for gradient clipping.
        """
        return self._clip_norm

    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        self._prepared = True
        return True

    def sanity_check(self):
        return self.optimizer in VALID_OPTIMIZERS and self.clip_norm > 0.0

    def to_dict(self):
        return {
            "optimizer": self.optimizer,
            "optimizer_params": self.optimizer_params,
            "use_gradient_clipping": self.use_gradient_clipping,
            "clip_norm": self.clip_norm,
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
