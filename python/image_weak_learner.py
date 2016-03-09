from __future__ import division

import math
import struct
import numpy as np

from statistics import HistogramStatistics


class WeakLearnerContext(object):

    def __init__(self, parameters, image_data):
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

    def compute_statistics(self, sample_indices):
        histogram = np.zeros((self._numOfLabels,), dtype=np.int64, order='C')
        for i in xrange(sample_indices.shape[0]):
            sample_index = sample_indices[i]
            label = self._flat_labels[sample_index]
            histogram[label] += 1
        statistics = HistogramStatistics.create_from_histogram_array(histogram)
        return statistics

    def sample_split_points(self, sample_indices, num_of_features, num_of_thresholds):
        return SplitPointCollection(self._parameters, num_of_features, num_of_thresholds)

    def compute_split_statistics(self, sample_indices, split_points):
        offsets = split_points._offsets
        thresholds = split_points._thresholds
        split_statistics = SplitStatistics(offsets.shape[0], thresholds.shape[1], self._numOfLabels)
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

    def select_best_split_point(self, current_statistics, split_statistics, return_information_gain=False):
        num_of_features = split_statistics._leftChildStatistics.shape[0]
        num_of_thresholds = split_statistics._leftChildStatistics.shape[1]
        current_histogram = current_statistics.histogram

        best_feature_id = 0
        best_threshold_id = 0
        best_information_gain = float('-inf')
        information_gain = float('-inf')
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

    def _compute_entropy(self, histogram, num_of_samples):
        ent = 0
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            if count > 0:
                relative_count = float(count) / num_of_samples
                ent -= relative_count * np.log2(relative_count)
        return ent

    def _compute_entropy2(self, statistics, num_of_samples, i, j):
        ent = 0
        for k in xrange(statistics.shape[2]):
            count = statistics[i, j, k]
            if count > 0:
                relative_count = float(count) / num_of_samples
                ent -= relative_count * np.log2(relative_count)
        return ent

    def _compute_num_of_samples(self, histogram):
        num_of_samples = 0
        for i in xrange(histogram.shape[0]):
            count = histogram[i]
            num_of_samples += count
        return num_of_samples

    def _compute_num_of_samples2(self, statistics, i, j):
        num_of_samples = 0
        for k in xrange(statistics.shape[2]):
            count = statistics[i, j, k]
            num_of_samples += count
        return num_of_samples

    def _compute_information_gain2(self, parent_histogram, left_child_statistics, right_child_statistics, i, j):
        parent_num_of_samples = self._compute_num_of_samples(parent_histogram)
        left_child_num_of_samples = self._compute_num_of_samples2(left_child_statistics, i, j)
        right_child_num_of_samples = self._compute_num_of_samples2(right_child_statistics, i, j)
        #print("parent - childs = {}".format(parent_num_of_samples - left_child_num_of_samples - right_child_num_of_samples))
        parent_entropy = self._compute_entropy(parent_histogram, parent_num_of_samples)
        left_child_entropy = self._compute_entropy2(left_child_statistics, left_child_num_of_samples, i, j)
        right_child_entropy = self._compute_entropy2(right_child_statistics, right_child_num_of_samples, i, j)
        #print("parent={}, left={}, right={}".format(parent_entropy, left_child_entropy, right_child_entropy))
        information_gain = parent_entropy \
            - (left_child_num_of_samples * left_child_entropy
               + right_child_num_of_samples * right_child_entropy) \
            / parent_num_of_samples
        return information_gain

    def _get_label(self, sample_index):
        return self._flat_labels[sample_index]

    # TODO: compute feature responses for list of sample indices and let ForestTrainer do the partitioning?
    def partition(self, sample_indices, split_point):
        # TODO: this is definitely not the most efficient way (explicit sorting is not necessary)
        offsets = split_point.feature
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

        threshold = split_point.threshold
        i_left = 0
        i_right = sample_indices.shape[0] - 1
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
        if feature_values_view[i_left] >= threshold:
            i_split = i_left
        else:
            i_split = i_left + 1

        # Optional: check partitioning
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


class Parameters:

    def __init__(self, feature_offset_range_low=0, feature_offset_range_high=10,
                  threshold_range_low=-1.0, threshold_range_high=+1.0):
        self._feature_offset_range_low = feature_offset_range_low
        self._feature_offset_range_high = feature_offset_range_high
        self._threshold_range_low = threshold_range_low
        self._threshold_range_high = threshold_range_high


class SplitPoint:

    SPLIT_POINT_FORMAT = '>4i1d'

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

    def to_array(self):
        return np.array(np.hstack([self._offsets, self._threshold]))


class SplitStatistics:

    def __init__(self, num_of_features, num_of_thresholds, num_of_labels):
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


class SplitPointCollection:

    def __init__(self, parameters, num_of_features, num_of_thresholds):
        # feature is a matrix with num_of_features rows and 4 columns for the offsets
        offset_range = np.hstack([np.arange(-parameters._feature_offset_range_high, -parameters._feature_offset_range_low + 1),
                                  np.arange(+parameters._feature_offset_range_low, +parameters._feature_offset_range_high + 1)])
        self._offsets = np.random.choice(offset_range, size=(num_of_features, 4))
        # self._offsets = np.random.random_integers(self.FEATURE_OFFSET_RANGE_LOW, +self.FEATURE_OFFSET_RANGE_HIGH,
        #                                          size=(num_of_features, 4))
        self._thresholds = np.random.uniform(parameters._threshold_range_low, parameters._threshold_range_high,
                                             size=(num_of_features, num_of_thresholds))

    def get_split_point(self, split_point_id):
        feature_id, threshold_id = split_point_id
        return SplitPoint(self._offsets[feature_id, :], self._thresholds[feature_id, threshold_id])


class FeatureEvaluator:

    def __init__(self, flat_data, image_width, image_height):
        self._image_width = image_width
        self._image_height = image_height
        self._image_stride = image_width * image_height
        self._flat_data = flat_data

    def compute_feature_value(self, sample_index, offsets):
        offset_x1 = offsets[0]
        offset_y1 = offsets[1]
        offset_x2 = offsets[2]
        offset_y2 = offsets[3]
        return self.compute_feature_value_with_offsets(sample_index, offset_x1, offset_y1, offset_x2, offset_y2)

    def compute_feature_value_with_offsets(self, sample_index,
                                              offset_x1, offset_y1,
                                            offset_x2, offset_y2):
        local_index = sample_index % (self._image_stride)
        #local_x = local_index // self._imageHeight
        #local_y = local_index % self._imageHeight

        offset1 = offset_x1 * self._image_height + offset_y1
        offset2 = offset_x2 * self._image_height + offset_y2

        if local_index + offset1 >= 0 and local_index + offset1 < self._image_stride:
            pixel1 = self._flat_data[sample_index + offset1]
        else:
            pixel1 = 0.0
        if local_index + offset2 >= 0 and local_index + offset2 < self._image_stride:
            pixel2 = self._flat_data[sample_index + offset2]
        else:
            pixel2 = 0.0

        return pixel1 - pixel2
