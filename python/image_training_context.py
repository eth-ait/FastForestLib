from __future__ import division

import numpy as np
import struct

from training_context import TrainingContext, SplitPointContext, SplitPoint
from statistics import HistogramStatistics


class SparseImageTrainingContext(TrainingContext):

    class _SplitPointContext(SplitPointContext):

        class _SplitPoint(SplitPoint):

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

            @staticmethod
            def from_array(array):
                assert len(array) == 5
                offsets = np.asarray(array[:4], dtype=np.int)
                threshold = np.asarray(array[-1], dtype=np.float)
                return SparseImageTrainingContext._SplitPointContext._SplitPoint(offsets, threshold)

        FEATURE_OFFSET_WINDOW = 25
        THRESHOLD_RANGE_LOW = -1000.0
        THRESHOLD_RANGE_HIGH = +1000.0

        def __init__(self, training_context, sample_indices, num_of_features, num_of_thresholds):
            self._trainingContext = training_context
            self._sampleIndices = sample_indices
            # feature is a matrix with num_of_features rows and 4 columns for the offsets
            self._features = np.random.random_integers(-self.FEATURE_OFFSET_WINDOW, +self.FEATURE_OFFSET_WINDOW,
                                                     size=(num_of_features, 4))
            self._thresholds = np.random.uniform(self.THRESHOLD_RANGE_LOW, self.THRESHOLD_RANGE_HIGH,
                                                 size=(num_of_features, num_of_thresholds))
            self._childStatistics = np.zeros(
                (2,
                 num_of_features,
                 num_of_thresholds,
                 training_context._numOfLabels),
                dtype=np.int, order='C')
            self._leftChildStatistics = self._childStatistics[0, :, :, :]
            self._rightChildStatistics = self._childStatistics[1, :, :, :]

        # TODO: compute feature responses for list of sample indices and let ForestTrainer compute the statistics??
        # TODO: computation of statistics for different thresholds can probably be optimized
        def compute_split_statistics(self):
            for i in xrange(self._features.shape[0]):
                feature = self._features[i, :]
                for sample_index in self._sampleIndices:
                    v = self._trainingContext.compute_feature_value(sample_index, feature)
                    l = self._trainingContext._get_label(sample_index)
                    for j in xrange(self._thresholds.shape[1]):
                        threshold = self._thresholds[i, j]
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
            best_feature_id = 0
            best_threshold_id = 0
            best_information_gain = -np.inf

            for i in xrange(self._features.shape[0]):
                for j in xrange(self._thresholds.shape[1]):
                    left_child_statistics = HistogramStatistics.create_from_histogram_array(self._leftChildStatistics[i, j, :])
                    right_child_statistics = HistogramStatistics.create_from_histogram_array(self._rightChildStatistics[i, j, :])
                    information_gain = self._trainingContext._compute_information_gain(
                        current_statistics, left_child_statistics, right_child_statistics)
                    if information_gain > best_information_gain:
                        best_feature_id = i
                        best_threshold_id = j
                        best_information_gain = information_gain
            best_split_point_id = (best_feature_id, best_threshold_id)

            if return_information_gain:
                return_value = (best_split_point_id, best_information_gain)
            else:
                return_value = best_split_point_id
            return return_value

        # TODO: compute feature responses for list of sample indices and let ForestTrainer do the partitioning?
        def partition(self, sample_indices, split_point_id):
            # TODO: this is definitely not the most efficient way (explicit sorting is not necessary)
            feature_id, threshold_id = split_point_id
            # compute feature values and sort the indices array according to them
            feature_values = np.empty((len(sample_indices),), dtype=np.float)
            assert isinstance(feature_values, np.ndarray)
            for i in xrange(len(feature_values)):
                sample_index = sample_indices[i]
                feature_values[i] = self._trainingContext.compute_feature_value(sample_index,
                                                                                self._features[feature_id, :])
            sorted_indices = np.argsort(feature_values)
            feature_values = feature_values[sorted_indices]
            sample_indices[:] = sample_indices[sorted_indices]

            # find index that partitions the samples into left and right parts
            threshold = self._thresholds[feature_id, threshold_id]
            i_split = 0
            while i_split < len(sample_indices) and feature_values[i_split] < threshold:
                i_split += 1
            return i_split

        def get_split_point(self, split_point_id):
            feature_id, threshold_id = split_point_id
            return self._SplitPoint(self._features[feature_id, :], self._thresholds[feature_id, threshold_id])


    def __init__(self, image_data):
        self._imageData = image_data
        self._numOfLabels = self._compute_num_of_labels(image_data)
        self._statisticsBins = np.arange(self._numOfLabels + 1)

    def _compute_num_of_labels(self, image_data):
        labels = image_data.flat_labels
        unique_labels = np.unique(labels)
        return np.sum(unique_labels >= 0)

    def compute_statistics(self, sample_indices):
        labels = self._imageData.flat_labels[sample_indices]
        hist, bin_edges = np.histogram(labels, bins=self._statisticsBins)
        statistics = HistogramStatistics.create_from_histogram_array(hist)
        return statistics

    def sample_split_points(self, sample_indices, num_of_features, num_of_thresholds):
        return self._SplitPointContext(self, sample_indices, num_of_features, num_of_thresholds)

    def _compute_information_gain(self, parent_statistics, left_child_statistics, right_child_statistics):
        information_gain = parent_statistics.entropy() \
            - (left_child_statistics.num_of_samples * left_child_statistics.entropy()
               + right_child_statistics.num_of_samples * right_child_statistics.entropy()) \
            / parent_statistics.num_of_samples
        return information_gain

    def _get_label(self, sample_index):
        return self._imageData.flat_labels[sample_index]

    def compute_feature_values(self, sample_indices, feature):
        pass

    def compute_feature_value(self, sample_index, feature):
        image_index = sample_index // (self._imageData.image_width * self._imageData.image_height)
        local_index = sample_index % (self._imageData.image_width * self._imageData.image_height)
        local_x = local_index // self._imageData.image_height
        local_y = local_index % self._imageData.image_height
        offset_x1, offset_y1, offset_x2, offset_y2 = feature

        if local_x + offset_x1 < 0 or local_x + offset_x1 >= self._imageData.image_width \
                or local_y + offset_y1 < 0 or local_y + offset_y1 >= self._imageData.image_height:
            pixel1 = 0
        else:
            pixel1 = self._imageData.data[image_index, local_x + offset_x1, local_y + offset_y1]

        if local_x + offset_x2 < 0 or local_x + offset_x2 >= self._imageData.image_width \
                or local_y + offset_y2 < 0 or local_y + offset_y2 >= self._imageData.image_height:
            pixel2 = 0
        else:
            pixel2 = self._imageData.data[image_index, local_x + offset_x2, local_y + offset_y2]

        return pixel1 - pixel2
