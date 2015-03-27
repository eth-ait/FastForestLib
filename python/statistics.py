from __future__ import division

import numpy as np
from math import log
from abc import ABCMeta, abstractmethod, abstractproperty


class Statistics(object):
    """
    This is an interface for statistics classes that can be used for training random forests.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def num_of_samples(self):
        """
        @return: The number of samples that contributed to this statistics object.
        """
        pass

    @abstractmethod
    def entropy(self):
        """
        @return: Returns the (Shannon) entropy or differential entropy of this statistics object.
        """
        pass


class HistogramStatistics(Statistics):
    """
    This class gives a histogram over the labels of samples.
    """

    def __init__(self):
        """
        @return: An uninitialized object. Use L{create_from_histogram_array} to create new L{HistogramStatistics} objects.
        """
        self._histogram = None
        self._num_of_samples = None

    @staticmethod
    def create_from_histogram_array(histogram):
        """
        A new L{HistogramStatistics} object from an existing histogram array.
        @param histogram: The histogram over the labels as a L{numpy} array. The sum over the histogram should
                          reflect the number of samples that contributed to the histogram.
        @return: The new L{HistogramStatistics} object.
        """
        statistics = HistogramStatistics()
        statistics._histogram = histogram
        statistics._num_of_samples = np.sum(histogram)
        return statistics

    @property
    def num_of_samples(self):
        """
        @return: The number of samples that contributed to the histogram.
        """
        return self._num_of_samples

    @property
    def histogram(self):
        """
        @return: The histogram as a L{numpy} array
        """
        return self._histogram

    def entropy(self):
        """
        @return: The Shannon entropy of the histogram
        """
        ent = 0
        for i in xrange(len(self._histogram)):
            count = self._histogram[i]
            if count > 0:
                relative_count = count / self._num_of_samples
                ent -= relative_count * log(relative_count, 2)
        return ent
