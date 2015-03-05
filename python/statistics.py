from __future__ import division

import numpy as np
from math import log
from abc import ABCMeta, abstractmethod, abstractproperty


class Statistics(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def num_of_samples(self): pass

    @abstractmethod
    def entropy(self): pass


class HistogramStatistics(Statistics):

    def __init__(self):
        self._histogram = None
        self._num_of_samples = None

    @staticmethod
    def create_from_histogram_array(histogram):
        statistics = HistogramStatistics()
        statistics._histogram = histogram
        statistics._num_of_samples = np.sum(histogram)
        return statistics

    @property
    def num_of_samples(self):
        return self._num_of_samples

    @property
    def histogram(self):
        return self._histogram

    def entropy(self):
        ent = 0
        for i in xrange(len(self._histogram)):
            count = self._histogram[i]
            if count > 0:
                relative_count = count / self._num_of_samples
                ent -= relative_count * log(relative_count, 2)
        return ent
