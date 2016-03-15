# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
"""# cython: boundscheck=True, wraparound=True, nonecheck=True, initializedcheck=True, cdivision=False"""
# cython: profile=True

from __future__ import division

cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from numpy.math cimport INFINITY
import struct
from libc.math cimport log2

from statistics import HistogramStatistics


cdef class SplitPoint:

    SPLIT_POINT_FORMAT = '>4i1d'

    cdef np.int64_t[::1] _offsets
    cdef np.float64_t _threshold

    def __cinit__(self, np.int64_t[::1] offsets, np.float64_t threshold):
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
        num_of_bytes = struct.calcsize(WeakLearnerContext._SplitPointContext._SplitPoint.SPLIT_POINT_FORMAT)
        raw_bytes = stream.read(num_of_bytes)
        tup = struct.unpack(WeakLearnerContext._SplitPointContext._SplitPoint.SPLIT_POINT_FORMAT, raw_bytes)
        offsets = np.array(tup[:4])
        threshold = tup[5]
        return WeakLearnerContext._SplitPointContext._SplitPoint(offsets, threshold)

    def to_array(self):
        return np.array(np.hstack([self._offsets, self._threshold]))

    @cython.wraparound(True)
    @staticmethod
    def from_array(array):
        assert(len(array) == 3)
        offsets = np.asarray(array[:4], dtype=np.int)
        threshold = np.asarray(array[-1], dtype=np.float)
        return SplitPointCollection._SplitPoint(offsets, threshold)


cdef class SplitStatistics:

    cdef np.int64_t[:, :, :, ::1] _childStatistics
    cdef np.int64_t[:, :, ::1] _leftChildStatistics
    cdef np.int64_t[:, :, ::1] _rightChildStatistics

    def __cinit__(self, int num_of_features, int num_of_thresholds, int num_of_labels):
        self._childStatistics = np.zeros(
            (2,
             num_of_features,
             num_of_thresholds,
             num_of_labels),
            dtype=np.int64, order='C')
        self._leftChildStatistics = self._childStatistics[0, :, :, :]
        self._rightChildStatistics = self._childStatistics[1, :, :, :]

    def get_buffer(self):
        return self._childStatistics

    def accumulate(self, statistics):
        self._childStatistics += statistics

cdef class Parameters:

    cdef np.int64_t _feature_offset_range_low
    cdef np.int64_t _feature_offset_range_high
    cdef np.float64_t _threshold_range_low
    cdef np.float64_t _threshold_range_high

    def __cinit__(self, feature_offset_range_low=0, feature_offset_range_high=10,
                  threshold_range_low=-1.0, threshold_range_high=+1.0):
        self._feature_offset_range_low = feature_offset_range_low
        self._feature_offset_range_high = feature_offset_range_high
        self._threshold_range_low = threshold_range_low
        self._threshold_range_high = threshold_range_high


cdef class SplitPointCollection:

    cdef np.int64_t[:, ::1] _offsets
    cdef np.float64_t[:, ::1] _thresholds

    def __cinit__(self, Parameters parameters, int num_of_features, int num_of_thresholds):
        # feature is a matrix with num_of_features rows and 4 columns for the offsets
        offset_range = np.hstack([np.arange(-parameters._feature_offset_range_high, -parameters._feature_offset_range_low + 1),
                                  np.arange(+parameters._feature_offset_range_low, +parameters._feature_offset_range_high + 1)])
        self._offsets = np.random.choice(offset_range, size=(num_of_features, 4))
        # self._offsets = np.random.random_integers(self.FEATURE_OFFSET_RANGE_LOW, +self.FEATURE_OFFSET_RANGE_HIGH,
        #                                          size=(num_of_features, 4))
        self._thresholds = np.random.uniform(parameters._threshold_range_low, parameters._threshold_range_high,
                                             size=(num_of_features, num_of_thresholds))

    def get_split_point(self, split_point_id):
        cdef int feature_id, threshold_id
        feature_id, threshold_id = split_point_id
        return SplitPoint(self._offsets[feature_id, :], self._thresholds[feature_id, threshold_id])


cdef class WeakLearnerContext:

    cdef FeatureEvaluator _feature_evaluator
    cdef Parameters _parameters
    cdef _imageData
    cdef _numOfLabels
    cdef _statisticsBins
    cdef int _imageWidth, _imageHeight, _imageStride
    cdef np.float64_t[:, :, ::1] _data
    cdef np.float64_t[::1] _flat_data
    cdef np.int64_t[:, :, ::1] _labels
    cdef np.int64_t[::1] _flat_labels

    def __init__(self, Parameters parameters, image_data):
        self._feature_evaluator = FeatureEvaluator(
            image_data.flat_data, image_data.image_width, image_data.image_height)
        self._parameters = parameters
        self._imageData = image_data
        self._numOfLabels = image_data.num_of_labels
        self._statisticsBins = np.arange(self._numOfLabels + 1)
        self._imageWidth = image_data.image_width
        self._imageHeight = image_data.image_height
        self._imageStride = self._imageWidth * self._imageHeight
        self._data = image_data.data
        self._labels = image_data.labels
        self._flat_data = image_data.flat_data
        self._flat_labels = image_data.flat_labels

    # TODO
    def get_num_of_labels(self):
        return self._numOfLabels

    def compute_statistics(self, np.int64_t[::1] sample_indices not None):
        cdef np.int64_t[::1] histogram = np.zeros((self._numOfLabels,), dtype=np.int64, order='C')
        cdef int i
        cdef np.int64_t sample_index, label
        for i in xrange(sample_indices.shape[0]):
            sample_index = sample_indices[i]
            label = self._flat_labels[sample_index]
            histogram[label] += 1
        statistics = HistogramStatistics.create_from_histogram_array(histogram)
        return statistics

    def sample_split_points(self, np.int64_t[::1] sample_indices not None, int num_of_features, int num_of_thresholds):
        return SplitPointCollection(self._parameters, num_of_features, num_of_thresholds)

    # TODO: compute feature responses for list of sample indices and let ForestTrainer compute the statistics??
    # TODO: computation of statistics for different thresholds can probably be optimized
    def compute_split_statistics(self, np.int64_t[::1] sample_indices not None, SplitPointCollection split_points):
        cdef np.int64_t[:, ::1] offsets = split_points._offsets
        cdef np.float64_t[:, ::1] thresholds = split_points._thresholds
        cdef SplitStatistics split_statistics = SplitStatistics(offsets.shape[0], thresholds.shape[1], self._numOfLabels)
        cdef int i, j, l, k
        cdef np.int64_t sample_index
        cdef double v, threshold
        #cdef int offset1, offset2
        cdef int offset_x1, offset_y1, offset_x2, offset_y2
        #for i in xrange(offsets.shape[0]):
        # TODO
        #with nogil, parallel():
            #for i in prange(offsets.shape[0]):
        for i in xrange(offsets.shape[0]):
            offset_x1 = offsets[i, 0]
            offset_y1 = offsets[i, 1]
            offset_x2 = offsets[i, 2]
            offset_y2 = offsets[i, 3]
            for k in xrange(sample_indices.shape[0]):
                sample_index = sample_indices[k]
                v = self._feature_evaluator.compute_feature_value_with_offsets(sample_index, offset_x1, offset_y1, offset_x2, offset_y2)
                l = self._get_label(sample_index)
                for j in xrange(thresholds.shape[1]):
                    threshold = thresholds[i, j]
                    if v < threshold:
                        split_statistics._leftChildStatistics[i, j, l] += 1
                    else:
                        split_statistics._rightChildStatistics[i, j, l] += 1
        return split_statistics

    # TODO: compute information gain for all features and thresholds instead and let ForestTrainer do the selection?
    def select_best_split_point(self, current_statistics, SplitStatistics split_statistics, return_information_gain=False):
        cdef int num_of_features = split_statistics._leftChildStatistics.shape[0]
        cdef int num_of_thresholds = split_statistics._leftChildStatistics.shape[1]
        cdef np.int64_t[::1] current_histogram = current_statistics.histogram

        cdef int best_feature_id = 0
        cdef int best_threshold_id = 0
        cdef double best_information_gain = -INFINITY
        cdef double information_gain
        cdef int i, j
        for i in xrange(num_of_features):
            for j in xrange(num_of_thresholds):
                information_gain = self._compute_information_gain2(
                    current_histogram, split_statistics._leftChildStatistics, split_statistics._rightChildStatistics, i, j)
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
    def partition(self, np.ndarray[np.int64_t, ndim=1, mode='c'] sample_indices not None, SplitPoint split_point):
        # TODO: this is definitely not the most efficient way (explicit sorting is not necessary)
        cdef int i
        cdef np.int64_t sample_index
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] feature_values
        cdef np.float64_t[::1] feature_values_view
        cdef int offset_x1, offset_y1, offset_x2, offset_y2
        cdef np.int64_t[::1] offsets = split_point.feature
        # compute feature values and sort the indices array according to them
        feature_values = np.empty((sample_indices.shape[0],), dtype=np.float)
        feature_values_view = feature_values
        #assert(isinstance(feature_values, np.ndarray))
        for i in xrange(sample_indices.shape[0]):
            sample_index = sample_indices[i]
            #feature_values_view[i] = self._trainingContext._compute_feature_value(sample_index,
            #                                                                offset1, offset2)
            feature_values_view[i] = self._feature_evaluator.compute_feature_value(sample_index, offsets)
            #print("feature_values[{}]={}".format(i, feature_values_view[i]))

        cdef np.float64_t threshold
        threshold = split_point.threshold
        cdef int i_left = 0
        cdef int i_right = sample_indices.shape[0] - 1
        cdef np.float64_t value
        while i_right > i_left:
            if feature_values_view[i_left] >= threshold:
                # swap feature values and sample indices
                value = feature_values_view[i_left]
                sample_index = sample_indices[i_left]
                feature_values_view[i_left] = feature_values_view[i_right]
                sample_indices[i_left] = sample_indices[i_right]
                feature_values_view[i_right] = value
                sample_indices[i_right] = sample_index

                i_right -= 1
            else:
                i_left += 1
        cdef int i_split
        if feature_values_view[i_left] >= threshold:
            i_split = i_left
        else:
            i_split = i_left + 1

        # check partitioning
        # for i in xrange(i_split):
        #     sample_index = sample_indices[i]
        #     value = self._compute_feature_value(sample_index,
        #                                         offset_x1, offset_y1,
        #                                         offset_x2, offset_y2)
        #     assert(value < threshold)
        # for i in xrange(i_split, sample_indices.shape[0]):
        #     sample_index = sample_indices[i]
        #     value = self._compute_feature_value(sample_index,
        #                                         offset_x1, offset_y1,
        #                                         offset_x2, offset_y2)
        #     assert(value >= threshold)

        return i_split

    @cython.profile(False)
    cdef double _compute_entropy(self, np.int64_t[::1] histogram, int num_of_samples) nogil:
        cdef double ent = 0, relative_count
        cdef int i, count
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            if count > 0:
                relative_count = (<double>count) / num_of_samples
                ent -= relative_count * log2(relative_count)
        return ent

    @cython.profile(False)
    cdef double _compute_entropy2(self, np.int64_t[:, :, ::1] statistics, int num_of_samples,
                                  int i, int j) nogil:
        cdef double ent = 0, relative_count
        cdef int k, count
        for k in xrange(statistics.shape[2]):
            count = statistics[i, j, k]
            if count > 0:
                relative_count = (<double>count) / num_of_samples
                ent -= relative_count * log2(relative_count)
        return ent

    @cython.profile(False)
    cdef inline int _compute_num_of_samples(self, np.int64_t[::1] histogram) nogil:
        cdef int num_of_samples = 0, count, k
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            num_of_samples += count
        return num_of_samples

    @cython.profile(False)
    cdef inline int _compute_num_of_samples2(self, np.int64_t[:, :, ::1] statistics,
                                             int i, int j) nogil:
        cdef int num_of_samples = 0, count, k
        for k in xrange(statistics.shape[2]):
            count = statistics[i, j, k]
            num_of_samples += count
        return num_of_samples

    @cython.profile(False)
    cdef inline double _compute_information_gain(self,
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

    @cython.profile(False)
    cdef inline double _compute_information_gain2(self,
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

    @cython.profile(False)
    cdef inline int _get_label(self, np.int64_t sample_index) nogil:
        return self._flat_labels[sample_index]

    def compute_feature_value(self, np.int64_t sample_index, np.int64_t[::1] feature not None):
        # cdef int offset1 = feature[0]
        # cdef int offset2 = feature[1]
        cdef np.int64_t[::1] offsets = feature
        return self._feature_evaluator.compute_feature_value(sample_index, offsets)


cdef class FeatureEvaluator:

    cdef np.int64_t _image_stride
    cdef np.int64_t _image_width
    cdef np.int64_t _image_height
    cdef np.float64_t[::1] _flat_data

    def __cinit__(self, np.float64_t[::1] flat_data, np.int64_t image_width, np.int64_t image_height):
        self._image_width = image_width
        self._image_height = image_height
        self._image_stride = image_width * image_height
        self._flat_data = flat_data

    # TODO
    #@cython.cdivision(True)
    #@cython.profile(False)
    #cdef inline double compute_feature_value(self, np.int64_t sample_index, np.int64_t[::1] offsets) nogil:
    @cython.cdivision(True)
    @cython.profile(False)
    cdef inline double compute_feature_value(self, np.int64_t sample_index, np.int64_t[::1] offsets):
        cdef np.int64_t offset_x1 = offsets[0]
        cdef np.int64_t offset_y1 = offsets[1]
        cdef np.int64_t offset_x2 = offsets[2]
        cdef np.int64_t offset_y2 = offsets[3]
        return self.compute_feature_value_with_offsets(sample_index, offset_x1, offset_y1, offset_x2, offset_y2)

    # TODO
    #@cython.cdivision(True)
    #@cython.profile(False)
    #cdef inline double compute_feature_value_with_offsets(self, np.int64_t sample_index,
    #                                                      np.int64_t offset_x1, np.int64_t offset_y1,
    #                                                      np.int64_t offset_x2, np.int64_t offset_y2) nogil:
    @cython.cdivision(True)
    @cython.profile(False)
    cdef inline double compute_feature_value_with_offsets(self, np.int64_t sample_index,
                                                          np.int64_t offset_x1, np.int64_t offset_y1,
                                                          np.int64_t offset_x2, np.int64_t offset_y2):
        cdef np.int64_t local_index
        local_index = sample_index % (self._image_stride)
        cdef int local_x, local_y
        local_x = local_index // self._image_height
        local_y = local_index % self._image_height

        cdef int x1, y1, x2, y2
        x1 = local_x + offset_x1
        y1 = local_y + offset_y1
        x2 = local_x + offset_x2
        y2 = local_y + offset_y2

        cdef np.int64_t offset1 = offset_x1 * self._image_height + offset_y1
        cdef np.int64_t offset2 = offset_x2 * self._image_height + offset_y2

        cdef np.float64_t pixel1, pixel2
        #if local_index + offset1 >= 0 and local_index + offset1 < self._image_stride:
        #    pixel1 = self._flat_data[sample_index + offset1]
        #    print("pixel1 at [" + str((sample_index + offset1) % 64) + "," + str((sample_index + offset1) / 64) + "] [" + str(sample_index + offset1) + "]: " + str(pixel1))
        #else:
        #    pixel1 = 0.0
        #    print("pixel1 out of bounds: " + str(pixel1))
        #if local_index + offset2 >= 0 and local_index + offset2 < self._image_stride:
        #    pixel2 = self._flat_data[sample_index + offset2]
        #    print("pixel2 at [" + str((sample_index + offset2) % 64) + "," + str((sample_index + offset2) / 64) + "] [" + str(sample_index + offset2) + "]: " + str(pixel2))
        #else:
        #    pixel2 = 0.0
        #    print("pixel2 out of bounds: " + str(pixel2))
        if x1 < 0 or x1 >= self._image_width or y1 < 0 or y1 >= self._image_height:
            pixel1 = 0.0
            #print("pixel1 out of bounds: " + str(pixel1))
        else:
            pixel1 = self._flat_data[sample_index + offset1]
            #print("pixel1 at [" + str((sample_index + offset1) % 64) + "," + str((sample_index + offset1) / 64) + "] [" + str(sample_index + offset1) + "]: " + str(pixel1))
        if x2 < 0 or x2 >= self._image_width or y2 < 0 or y2 >= self._image_height:
            pixel2 = 0.0
            #print("pixel2 out of bounds: " + str(pixel2))
        else:
            pixel2 = self._flat_data[sample_index + offset2]
            #print("pixel2 at [" + str((sample_index + offset2) % 64) + "," + str((sample_index + offset2) / 64) + "] [" + str(sample_index + offset2) + "]: " + str(pixel2))

        return pixel1 - pixel2


cdef class Predictor:

    @staticmethod
    def read_from_matlab_file(forest_file):
        import scipy.io
        m_dict = scipy.io.loadmat(forest_file)
        tree_matrices = []
        forest = m_dict['forest']
        for i in xrange(forest.shape[0]):
            for j in xrange(forest.shape[1]):
                tree_matrices.append(np.ascontiguousarray(forest[i, j], dtype=np.float64))
        return Predictor(tree_matrices)

    cdef _tree_matrices

    def __cinit__(self, tree_matrices):
        self._tree_matrices = tree_matrices

    cdef _find_node_recursive(self, int node_index, np.float64_t[:, ::1] tree_matrix, np.int64_t sample_index,
                              FeatureEvaluator evaluator, int stop_node_index):
        cdef np.float64_t[::1] offsets
        cdef np.int64_t offset_x1, offset_y1, offset_x2, offset_y2
        cdef np.float64_t threshold
        cdef np.float64_t value
        cdef int child_node_index
        cdef int leaf_indicator = <int>tree_matrix[node_index, tree_matrix.shape[1] - 1]
        if leaf_indicator == 1 or node_index >= stop_node_index:
            return node_index
        else:
            offsets = tree_matrix[node_index, :4]
            offset_x1 = <np.int64_t>offsets[0]
            offset_y1 = <np.int64_t>offsets[1]
            offset_x2 = <np.int64_t>offsets[2]
            offset_y2 = <np.int64_t>offsets[3]
            threshold = tree_matrix[node_index, 4]
            #print('node_index: ' + str(node_index))
            #print('threshold: ' + str(threshold))
            #print('offset_x1: ' + str(offset_x1))
            #print('offset_y1: ' + str(offset_y1))
            #print('offset_x2: ' + str(offset_x2))
            #print('offset_y2: ' + str(offset_y2))
            value = evaluator.compute_feature_value_with_offsets(sample_index, offset_x1, offset_y1, offset_x2, offset_y2)
            #print("pixel_difference: " + str(value))
            #value = SparseImageFeatureEvaluator.compute_image_feature_value(sample_index, image, offsets)
            if value < threshold:
                #print("left")
                child_node_index = 2 * node_index + 1
            else:
                #print("right")
                child_node_index = 2 * node_index + 2
            return self._find_node_recursive(child_node_index, tree_matrix, sample_index,
                                             evaluator, stop_node_index)

    # cdef _find_node_recursive(self, int node_index, np.float64_t[:, ::1] tree_matrix, np.int64_t sample_index, np.float64_t[:, ::1] image):
    #     cdef np.float64_t[::1] offsets
    #     cdef np.float64_t threshold
    #     cdef np.float64_t value
    #     cdef int child_node_index
    #     cdef int leaf_indicator = <int>tree_matrix[node_index, tree_matrix.shape[1] - 1]
    #     if leaf_indicator == 1:
    #         return node_index
    #     else:
    #         offsets = tree_matrix[node_index, :4]
    #         threshold = tree_matrix[node_index, 4]
    #         value = SparseImageFeatureEvaluator.compute_image_feature_value(sample_index, image, offsets)
    #         if value < threshold:
    #             child_node_index = 2 * node_index + 1
    #         else:
    #             child_node_index = 2 * node_index + 2
    #         return self._find_node_recursive(child_node_index, tree_matrix, sample_index, image)

    cdef _find_node(self, np.float64_t[:, ::1] tree_matrix, np.int64_t sample_index, FeatureEvaluator evaluator,
                    int stop_node_index):
        return self._find_node_recursive(0, tree_matrix, sample_index, evaluator, stop_node_index)

    # cdef _find_node(self, np.float64_t[:, ::1] tree_matrix, np.int64_t sample_index, np.float64_t[:, ::1] image):
    #     return self._find_node_recursive(0, tree_matrix, sample_index, image)

    def predict_image_aggregate_statistics(self, np.int64_t[::1] sample_indices, np.float64_t[:, ::1] image,
                                           int max_evaluation_depth=-1):
        cdef np.float64_t[::1] flat_image = np.array(image).reshape((image.size,), order='C')
        return self.predict_image_aggregate_statistics_with_flat_image(
            sample_indices, flat_image, image.shape[0], image.shape[1],
            max_evaluation_depth)

    def predict_image_aggregate_statistics_with_flat_image(self, np.int64_t[::1] sample_indices,
                                                           np.float64_t[::1] flat_image,
                                                           int image_width, int image_height,
                                                           int max_evaluation_depth=-1):
        cdef int num_of_samples = sample_indices.shape[0]
        cdef int num_of_labels = self._tree_matrices[0].shape[1] - 6
        cdef int node_index, j, k
        cdef int stop_node_index
        cdef np.int64_t sample_index
        cdef np.float64_t[:, ::1] tree_matrix
        cdef np.int64_t[:, ::1] aggregate_histogram = np.zeros((num_of_samples, num_of_labels), dtype=np.int64)
        cdef FeatureEvaluator evaluator = FeatureEvaluator(flat_image, image_width, image_height)
        for tree_matrix in self._tree_matrices:
            if max_evaluation_depth < 0:
                stop_node_index = tree_matrix.shape[0]
            else:
                stop_node_index = 2 ** (max_evaluation_depth - 1) - 1
            for k in xrange(num_of_samples):
                sample_index = sample_indices[k]
                node_index = self._find_node(tree_matrix, sample_index, evaluator, stop_node_index)
                #print('node_index: ' + str(node_index))
                #print('histogram: ' + str(np.array(tree_matrix[node_index, 5:-1])))
                for j in xrange(num_of_labels):
                    aggregate_histogram[k, j] += <np.int64_t>tree_matrix[node_index, 5 + j]
        return HistogramStatistics.create_from_histogram_array(aggregate_histogram)
