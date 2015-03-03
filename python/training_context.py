from __future__ import division

from abc import ABCMeta, abstractmethod


class TrainingContext:
    __metaclass__ = ABCMeta

    # TODO: abstract away the computation of statistics
    @abstractmethod
    def compute_statistics(self, sample_indices): pass

    @abstractmethod
    def sample_split_points(self, sample_indices, num_of_features, num_of_thresholds): pass

    @abstractmethod
    def compute_split_statistics(self, split_points): pass

    """Returns an ID that uniquely identifies the best split point"""
    @abstractmethod
    def select_best_split_point(self, current_statistics, split_statistics, return_information_gain=True): pass

    """Partitions the array sample_indices into a left and right part based on the specified split point.
    The returned index i_split is the partition point, i.e. sample_indices[:i_split] is the left part
    and sample_indices[i_split:] is the right part."""
    @abstractmethod
    def partition(self, sample_indices, best_split_point): pass


class SplitPointCollection:
    __metaclass__ = ABCMeta

    """Reconstruct a collection of split points from a buffer."""
    @staticmethod
    @abstractmethod
    def from_buffer(buf): pass

    """Return a buffer that represents the collection of split points."""
    @abstractmethod
    def get_buffer(self): pass

    """Return an object representing the specified split point."""
    def get_split_point(self, split_point_id): pass


class SplitStatistics:
    __metaclass__ = ABCMeta

    """Reconstruct statistics from a buffer."""
    @staticmethod
    @abstractmethod
    def from_buffer(buf): pass

    """Return a buffer that represents the statistics."""
    @abstractmethod
    def get_buffer(self): pass

    @abstractmethod
    def accumulate(self, statistics): pass


class SplitPoint:
    __metaclass__ = ABCMeta

    @abstractmethod
    def write(self, stream): pass

    @staticmethod
    @abstractmethod
    def read_from(stream): pass

    @abstractmethod
    def to_array(self): pass

    @staticmethod
    @abstractmethod
    def from_array(array): pass
