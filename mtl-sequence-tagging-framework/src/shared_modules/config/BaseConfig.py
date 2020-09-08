"""Abstract class for configurations."""

from abc import ABCMeta, abstractmethod, abstractproperty


class BaseConfig(object):
    """Abstract class for configurations.

    All configuration classes need to inherit from this base class.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def prepare(self):
        """
        Fill all properties not already populated at initialization with values.
        Returns:
            True in case of success, False otherwise.
        """
        raise NotImplementedError("Must define `prepare` to use this base class.")

    @abstractmethod
    def sanity_check(self):
        """
        Check if the properties of the configuration object are valid.
        Returns:
            True in case of validity, False otherwise.
        """
        raise NotImplementedError("Must define `sanity_check` to use this base class.")

    @abstractmethod
    def to_dict(self):
        """
        Convert configuration to a dictionary.
        Returns:
            Dictionary with all the information of the instance.
        """
        raise NotImplementedError("Must define `todict` to use this base class")

    @abstractmethod
    def set_paths(self, paths):
        """
        Set the paths for the experiment.
        Args:
            paths (`dict` of str): Necessary paths.
        """
        raise NotImplementedError("Must define `set_paths` to use this base class")

    @abstractproperty
    def paths_set(self):
        """
        Check if the paths have been set. Should be true after `set_paths` has been called.
        Returns:
            bool: True if the paths have been set, False otherwise.
        """
        raise NotImplementedError("Must define `paths_set` property to use this base class")

    @abstractproperty
    def prepared(self):
        """
        Check if the configuration has been prepared. Should be true after `prepare` has been called.
        Returns:
            bool: True if the configuration has been prepared, False otherwise.
        """
        raise NotImplementedError("Must define `prepared` property to use this base class")
