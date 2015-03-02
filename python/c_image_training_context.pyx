# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
# cython: profile=True
"""# cython: boundscheck=True, wraparound=True, nonecheck=True, initializedcheck=True, cdivision=False"""

from __future__ import division

cimport cython
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
import struct
from libc.math cimport log2

from statistics import HistogramStatistics

from image_training_context import ImageData, ImageDataReader


cdef class SplitPoint:

    SPLIT_POINT_FORMAT = '>2i1d'

    cdef _offsets
    cdef _threshold

    def __init__(self, offsets, threshold):
        self._offsets = offsets
        self._threshold = threshold

    @property
    def feature(self):
        return self._offsets

    @property
    def threshold(self):
        return self._threshold

    def write(self, stream):
        raw_bytes = struct.pack(self.SPLIT_POINT_FORMAT, *(self._offsets + [self._threshold]))
        stream.write(raw_bytes)

    @staticmethod
    def read_from(stream):
        num_of_bytes = struct.calcsize(SparseImageTrainingContext._SplitPointContext._SplitPoint.SPLIT_POINT_FORMAT)
        raw_bytes = stream.read(num_of_bytes)
        tup = struct.unpack(SparseImageTrainingContext._SplitPointContext._SplitPoint.SPLIT_POINT_FORMAT, raw_bytes)
        offsets = np.array(tup[:4])
        threshold = tup[5]
        return SparseImageTrainingContext._SplitPointContext._SplitPoint(offsets, threshold)

    def to_array(self):
        return np.array(np.hstack([self._offsets, self._threshold]))

    @cython.wraparound(True)
    @staticmethod
    def from_array(array):
        assert len(array) == 3
        offsets = np.asarray(array[:2], dtype=np.int)
        threshold = np.asarray(array[-1], dtype=np.float)
        return SplitPointContext._SplitPoint(offsets, threshold)


cdef class SplitPointContext:

    FEATURE_OFFSET_WINDOW = 25
    THRESHOLD_RANGE_LOW = -1000.0
    THRESHOLD_RANGE_HIGH = +1000.0

    cdef SparseImageTrainingContext _trainingContext
    cdef np.ndarray _sampleIndices
    cdef np.ndarray _features
    cdef np.ndarray _thresholds
    # cdef np.ndarray[int, ndim=4, mode='c'] _childStatistics
    cdef np.ndarray _childStatistics
    cdef np.int64_t[:, :, ::1] _leftChildStatistics
    cdef np.int64_t[:, :, ::1] _rightChildStatistics

    cpdef get_child_statistics(self):
        return np.array(self._leftChildStatistics), np.array(self._rightChildStatistics)

    def __init__(self, training_context, sample_indices, num_of_features, num_of_thresholds):
        self._trainingContext = training_context
        self._sampleIndices = sample_indices
        # feature is a matrix with num_of_features rows and 2 columns for the offsets
        self._features = np.random.random_integers(-self.FEATURE_OFFSET_WINDOW, +self.FEATURE_OFFSET_WINDOW,
                                                 size=(num_of_features, 2))
        self._thresholds = np.random.uniform(self.THRESHOLD_RANGE_LOW, self.THRESHOLD_RANGE_HIGH,
                                             size=(num_of_features, num_of_thresholds))
        self._childStatistics = np.zeros(
            (2,
             num_of_features,
             num_of_thresholds,
             training_context.get_num_of_labels()),
            dtype=np.int64, order='C')
        self._leftChildStatistics = self._childStatistics[0, :, :, :]
        self._rightChildStatistics = self._childStatistics[1, :, :, :]

    # TODO: compute feature responses for list of sample indices and let ForestTrainer compute the statistics??
    # TODO: computation of statistics for different thresholds can probably be optimized
    def compute_split_statistics(SplitPointContext self):
        cdef int i, j, l, k, sample_index
        cdef double v, threshold
        cdef np.int64_t[:, ::1] features = self._features
        cdef np.float64_t[:, ::1] thresholds = self._thresholds
        cdef np.int64_t[::1] sample_indices = self._sampleIndices
        cdef int offset1, offset2
        for i in xrange(features.shape[0]):
            offset1 = features[i, 0]
            offset2 = features[i, 1]
            for k in xrange(sample_indices.shape[0]):
                sample_index = sample_indices[k]
                v = self._trainingContext._compute_feature_value(sample_index, offset1, offset2)
                l = self._trainingContext._get_label(sample_index)
                for j in xrange(thresholds.shape[1]):
                    threshold = thresholds[i, j]
                    if v < threshold:
                        self._leftChildStatistics[i, j, l] += 1
                    else:
                        self._rightChildStatistics[i, j, l] += 1

    def get_split_statistics_buffer(self):
        return self._childStatistics

    def accumulate_split_statistics(self, statistics):
        self._childStatistics += statistics

    # TODO: compute information gain for all features and thresholds instead and let ForestTrainer do the selection?
    def select_best_split_point(self, current_statistics, return_information_gain=False):
        cdef int num_of_features = self._features.shape[0]
        cdef int num_of_thresholds = self._thresholds.shape[1]
        cdef np.int64_t[::1] current_histogram = current_statistics.histogram

        cdef int best_feature_id = 0
        cdef int best_threshold_id = 0
        cdef double best_information_gain = -INFINITY
        cdef double information_gain
        # remove unnecessary statements
        #cdef int num_of_labels = self._leftChildStatistics.shape[2]
        #cdef np.int64_t[::1] left_child_histogram = self._leftChildStatistics[0, 0, :]
        #cdef np.int64_t[::1] right_child_histogram = self._rightChildStatistics[0, 0, :]
        #cdef np.ndarray[np.int64_t, ndim=1, mode='c'] left_arr = np.empty((2,), dtype=np.int64)
        #cdef np.ndarray[np.int64_t, ndim=1, mode='c'] right_arr = np.empty((2,), dtype=np.int64)
        cdef int i, j
        for i in xrange(num_of_features):
            for j in xrange(num_of_thresholds):
                # left_child_histogram = self._leftChildStatistics[i, j, :]
                # right_child_histogram = self._rightChildStatistics[i, j, :]
                #information_gain = self._trainingContext._compute_information_gain(
                #    current_histogram, left_child_histogram, right_child_histogram)
                #left_arr.data = <char*>(&self._leftChildStatistics[i, j, 0])
                #right_arr.data = <char*>(&self._rightChildStatistics[i, j, 0])
                #information_gain = self._trainingContext._compute_information_gain3(
                #    current_histogram, left_arr, right_arr)
                information_gain = self._trainingContext._compute_information_gain2(
                    current_histogram, self._leftChildStatistics, self._rightChildStatistics, i, j)
                #print("information_gain={}".format(information_gain))
                #assert(information_gain >= 0)
                if information_gain > best_information_gain:
                    best_feature_id = i
                    best_threshold_id = j
                    best_information_gain = information_gain
                    #print("best_information_gain={}".format(best_information_gain))
        best_split_point_id = (best_feature_id, best_threshold_id)

        if return_information_gain:
            return_value = (best_split_point_id, best_information_gain)
        else:
            return_value = best_split_point_id
        return return_value

    # TODO: compute feature responses for list of sample indices and let ForestTrainer do the partitioning?
    def partition(self, np.ndarray[np.int64_t, ndim=1, mode='c'] sample_indices, tuple split_point_id):
        # TODO: this is definitely not the most efficient way (explicit sorting is not necessary)
        cdef int i, sample_index
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] feature_values
        cdef np.float64_t[::1] feature_values_view
        cdef int offset_x1, offset_y1, offset_x2, offset_y2
        feature_id, threshold_id = split_point_id
        offset1 = self._features[feature_id, 0]
        offset2 = self._features[feature_id, 1]
        # compute feature values and sort the indices array according to them
        feature_values = np.empty((sample_indices.shape[0],), dtype=np.float)
        feature_values_view = feature_values
        #assert isinstance(feature_values, np.ndarray)
        for i in xrange(sample_indices.shape[0]):
            sample_index = sample_indices[i]
            feature_values_view[i] = self._trainingContext._compute_feature_value(sample_index,
                                                                            offset1, offset2)
            #print("feature_values[{}]={}".format(i, feature_values_view[i]))
        sorted_indices = np.argsort(feature_values)
        feature_values[:] = feature_values[sorted_indices]
        sample_indices[:] = sample_indices[sorted_indices]

        # find index that partitions the samples into left and right parts
        cdef double threshold
        cdef int i_split
        threshold = self._thresholds[feature_id, threshold_id]
        #print("threshold={}".format(threshold))
        i_split = 0
        while i_split < sample_indices.shape[0] and feature_values_view[i_split] < threshold:
            #print("  i_split={}, value={}".format(i_split, feature_values_view[i_split]))
            i_split += 1
        return i_split

    def get_split_point(self, split_point_id):
        feature_id, threshold_id = split_point_id
        return SplitPoint(self._features[feature_id, :], self._thresholds[feature_id, threshold_id])


cdef class SparseImageTrainingContext:

    cdef _imageData
    cdef _numOfLabels
    cdef _statisticsBins
    cdef int _imageWidth, _imageHeight, _imageArea
    cdef np.float64_t[:, :, ::1] _data
    cdef np.float64_t[::1] _flat_data
    cdef np.int64_t[:, :, ::1] _labels
    cdef np.int64_t[::1] _flat_labels

    def __init__(self, image_data):
        self._imageData = image_data
        self._numOfLabels = self._compute_num_of_labels(image_data)
        self._statisticsBins = np.arange(self._numOfLabels + 1)
        self._imageWidth = image_data.image_width
        self._imageHeight = image_data.image_height
        self._imageArea = self._imageWidth * self._imageHeight
        self._data = image_data.data
        self._labels = image_data.labels
        self._flat_data = image_data.flat_data
        self._flat_labels = image_data.flat_labels

    def _compute_num_of_labels(self, image_data):
        labels = image_data.flat_labels
        unique_labels = np.unique(labels)
        return np.sum(unique_labels >= 0)

    # TODO
    def get_num_of_labels(self):
        return self._numOfLabels

    def compute_statistics(self, sample_indices):
        labels = self._imageData.flat_labels[sample_indices]
        hist, bin_edges = np.histogram(labels, bins=self._statisticsBins)
        statistics = HistogramStatistics.from_histogram_array(hist)
        return statistics

    def sample_split_points(self, sample_indices, num_of_features, num_of_thresholds):
        return SplitPointContext(self, sample_indices, num_of_features, num_of_thresholds)

    cdef double _compute_entropy(SparseImageTrainingContext self, np.int64_t[::1] histogram, int num_of_samples) nogil:
        cdef double ent = 0, relative_count
        cdef int i, count
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            if count > 0:
                relative_count = (<double>count) / num_of_samples
                ent -= relative_count * log2(relative_count)
        return ent

    # remove
    cdef double _compute_entropy2(SparseImageTrainingContext self, np.int64_t[:, :, ::1] statistics, int num_of_samples,
                                  int i, int j) nogil:
        cdef double ent = 0, relative_count
        cdef int k, count
        for k in xrange(statistics.shape[2]):
            count = statistics[i, j, k]
            if count > 0:
                relative_count = (<double>count) / num_of_samples
                ent -= relative_count * log2(relative_count)
        return ent

    # remove
    cdef double _compute_entropy3(SparseImageTrainingContext self, np.ndarray[np.int64_t, ndim=1, mode='c'] histogram, int num_of_samples):
        cdef double ent = 0, relative_count
        cdef int i, count
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            if count > 0:
                relative_count = (<double>count) / num_of_samples
                ent -= relative_count * log2(relative_count)
        return ent

    cdef inline int _compute_num_of_samples(SparseImageTrainingContext self, np.int64_t[::1] histogram) nogil:
        cdef int num_of_samples = 0, count, k
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            num_of_samples += count
        return num_of_samples

    # remove
    cdef inline int _compute_num_of_samples2(SparseImageTrainingContext self, np.int64_t[:, :, ::1] statistics,
                                             int i, int j) nogil:
        cdef int num_of_samples = 0, count, k
        for k in xrange(statistics.shape[2]):
            count = statistics[i, j, k]
            num_of_samples += count
        return num_of_samples

    # remove
    cdef int _compute_num_of_samples3(SparseImageTrainingContext self, np.ndarray[np.int64_t, ndim=1, mode='c'] histogram):
        cdef int num_of_samples = 0, count, k
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            num_of_samples += count
        return num_of_samples

    cdef inline double _compute_information_gain(SparseImageTrainingContext self,
                                                 np.int64_t[::1] parent_histogram,
                                                 np.int64_t[::1] left_child_histogram,
                                                 np.int64_t[::1] right_child_histogram) nogil:
        cdef int parent_num_of_samples = self._compute_num_of_samples(parent_histogram)
        cdef int left_child_num_of_samples = self._compute_num_of_samples(left_child_histogram)
        cdef int right_child_num_of_samples = self._compute_num_of_samples(right_child_histogram)
        #print("parent - childs = {}".format(parent_num_of_samples - left_child_num_of_samples - right_child_num_of_samples))
        cdef double parent_entropy = self._compute_entropy(parent_histogram, parent_num_of_samples)
        cdef double left_child_entropy = self._compute_entropy(left_child_histogram, left_child_num_of_samples)
        cdef double right_child_entropy = self._compute_entropy(right_child_histogram, right_child_num_of_samples)
        #print("parent={}, left={}, right={}".format(parent_entropy, left_child_entropy, right_child_entropy))
        information_gain = parent_entropy \
            - (left_child_num_of_samples * left_child_entropy
               + right_child_num_of_samples * right_child_entropy) \
            / parent_num_of_samples
        return information_gain

    # remove
    cdef inline double _compute_information_gain2(SparseImageTrainingContext self,
                                                 np.int64_t[::1] parent_histogram,
                                                 np.int64_t[:, :, ::1] left_child_statistics,
                                                 np.int64_t[:, :, ::1] right_child_statistics,
                                                 int i, int j) nogil:
        cdef int parent_num_of_samples = self._compute_num_of_samples(parent_histogram)
        cdef int left_child_num_of_samples = self._compute_num_of_samples2(left_child_statistics, i, j)
        cdef int right_child_num_of_samples = self._compute_num_of_samples2(right_child_statistics, i, j)
        #print("parent - childs = {}".format(parent_num_of_samples - left_child_num_of_samples - right_child_num_of_samples))
        cdef double parent_entropy = self._compute_entropy(parent_histogram, parent_num_of_samples)
        cdef double left_child_entropy = self._compute_entropy2(left_child_statistics, left_child_num_of_samples, i, j)
        cdef double right_child_entropy = self._compute_entropy2(right_child_statistics, right_child_num_of_samples, i, j)
        #print("parent={}, left={}, right={}".format(parent_entropy, left_child_entropy, right_child_entropy))
        information_gain = parent_entropy \
            - (left_child_num_of_samples * left_child_entropy
               + right_child_num_of_samples * right_child_entropy) \
            / parent_num_of_samples
        return information_gain

    # remove
    cdef double _compute_information_gain3(SparseImageTrainingContext self,
                                                 np.int64_t[::1] parent_histogram,
                                                 np.ndarray[np.int64_t, ndim=1, mode='c'] left_child_histogram,
                                                 np.ndarray[np.int64_t, ndim=1, mode='c'] right_child_histogram):
        cdef int parent_num_of_samples = self._compute_num_of_samples(parent_histogram)
        cdef int left_child_num_of_samples = self._compute_num_of_samples3(left_child_histogram)
        cdef int right_child_num_of_samples = self._compute_num_of_samples3(right_child_histogram)
        #print("parent - childs = {}".format(parent_num_of_samples - left_child_num_of_samples - right_child_num_of_samples))
        cdef double parent_entropy = self._compute_entropy(parent_histogram, parent_num_of_samples)
        cdef double left_child_entropy = self._compute_entropy3(left_child_histogram, left_child_num_of_samples)
        cdef double right_child_entropy = self._compute_entropy3(right_child_histogram, right_child_num_of_samples)
        #print("parent={}, left={}, right={}".format(parent_entropy, left_child_entropy, right_child_entropy))
        information_gain = parent_entropy \
            - (left_child_num_of_samples * left_child_entropy
               + right_child_num_of_samples * right_child_entropy) \
            / parent_num_of_samples
        return information_gain

    #@cython.profile(False)
    cdef inline int _get_label(SparseImageTrainingContext self, int sample_index) nogil:
        return self._flat_labels[sample_index]

    def compute_feature_value(self, int sample_index, np.int64_t[::1] feature):
        cdef int offset1 = feature[0]
        cdef int offset2 = feature[1]
        return self._compute_feature_value(sample_index, offset1, offset2)

    #@cython.profile(False)
    cdef inline double _compute_feature_value(SparseImageTrainingContext self, int sample_index,
                                             int offset1, int offset2) nogil:
        cdef int image_index, local_index
        image_index = sample_index // (self._imageArea)
        local_index = sample_index % (self._imageArea)
        #cdef int local_x, local_y
        #local_x = local_index // self._imageHeight
        #local_y = local_index % self._imageHeight

        cdef double pixel1, pixel2
        if local_index + offset1 >= 0 and local_index + offset1 < self._imageArea:
            pixel1 = self._flat_data[sample_index + offset1]
        else:
            pixel1 = 0.0
        if local_index + offset2 >= 0 and local_index + offset2 < self._imageArea:
            pixel2 = self._flat_data[sample_index + offset2]
        else:
            pixel2 = 0.0

        return pixel1 - pixel2
