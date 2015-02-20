import numpy as np

from forest import Forest
from tree import Tree


class TrainingParameters:

    def __init__(self, num_of_trees=3, maximum_depth=20, num_of_features=100,
                 num_of_thresholds=50, minimum_num_of_samples=100):
        self.num_of_trees = num_of_trees
        self.maximum_depth = maximum_depth
        self.num_of_features = num_of_features
        self.num_of_thresholds = num_of_thresholds
        self.minimum_num_of_samples = minimum_num_of_samples


class RandomForestTrainer:

    class RandomForestOperation:

        def __init__(self, training_context, training_parameters):
            self._trainingContext = training_context
            self._trainingParameters = training_parameters
            # TODO: make sure that split matrices are contigues C arrays
            # WARNING: statistics are only used temporary in the train_recursive method
            self._left_child_statistics = np.empty(
                (training_parameters.num_of_features,
                 training_parameters.num_of_thresholds,
                 training_context.num_of_labels),
                dtype=np.int, order='C')
            self._right_child_statistics = np.empty(
                (training_parameters.num_of_features,
                 training_parameters.num_of_thresholds,
                 training_context.num_of_labels),
                dtype=np.int, order='C')
            # self._sample_indices = np.arange((training_context.numf_of_samples(),), dtype=np.int, order='C')
            self._sample_indices = training_context.sample_indices

        # @property
        # def training_context(self):
        #     return self._trainingContext

        # @property
        # def data(self):
        #     return self._data

        # @property
        # def training_parameters(self):
        #     return self._trainingParameters

        def train_recursive(self, node, i_start, i_end, parent_statistics=None):

            # TODO: return sample statistics or assign to nodes

            if i_end - i_start < self._trainingParameters.minimum_num_of_samples:
                return

            # define local aliases for some long variable names
            left_child_statistics = self._left_child_statistics
            right_child_statistics = self._right_child_statistics
            sample_indices = self._sample_indices[i_start:i_end]

            if parent_statistics is None:
                parent_statistics = self._trainingContext.compute_statistics(sample_indices)

            # feature is a matrix with num_of_features rows and 4 columns for the offsets
            features = self._trainingContext.sample_random_features(self._trainingParameters.num_of_features)
            # thresholds is a matrix with num_of_features rows and num_of_thresholds columns
            thresholds = self._trainingContext.sample_random_thresholds(self._trainingParameters.num_of_features,
                                                                        self._trainingParameters.num_of_thresholds)

            # TODO: distribute features and thresholds to ranks > 0

            # TODO: abstract statistics structure into trainingContext
            # compute the statistics for all feature and threshold combinations
            for i, feature in enumerate(features):
                for j, threshold in enumerate(thresholds):
                    for sample_index in sample_indices:
                        v = self._trainingContext.compute_feature_value(sample_index, feature)
                        l = self._trainingContext.get_label(sample_index)
                        if v < threshold:
                            left_child_statistics[i, j, l] += 1
                        else:
                            right_child_statistics[i, j, l] += 1

            # TODO: send statistics to rank 0

            best_feature_index = 0
            best_threshold_index = 0
            best_information_gain = -np.inf

            # find the best feature
            for i, feature in enumerate(features):
                for j, threshold in enumerate(thresholds):
                    information_gain = self._trainingContext.compute_information_gain(
                        parent_statistics, left_child_statistics[i, j, :], right_child_statistics[i, j, :])
                    if information_gain < best_information_gain:
                        best_feature_index = i
                        best_threshold_index = j
                        best_information_gain = information_gain

            # TODO: send best feature and threshold to ranks > 0

            # compute feature values and sort the indices array according to them
            feature_values = np.empty((len(sample_indices),), dtype=np.float)
            assert isinstance(feature_values, np.ndarray)
            for i in xrange(len(feature_values)):
                sample_index = sample_indices[i]
                feature_values[i] = self._trainingContext.compute_feature_value(sample_index, features[best_feature_index])
            sorted_indices = np.argsort(feature_values)
            sample_indices[:] = sample_indices[sorted_indices]

            # find index that partitions the samples into left and right parts
            best_threshold = thresholds[best_feature_index, best_threshold_index]
            i_split = 0
            while i_split < len(sample_indices) and feature_values[i_split] < best_threshold:
                i_split += 1

            # train left and right child
            self.train_recursive(node.left_child, i_start, i_split, left_child_statistics)
            self.train_recursive(node.right_child, i_split + 1, i_end, right_child_statistics)

    @staticmethod
    def train_forest(training_context, training_parameters):
        forest = Forest()
        for i in xrange(training_parameters.num_of_trees):
            tree = RandomForestTrainer.train_tree(training_context, training_parameters)
            forest.append(tree)
        return forest

    @staticmethod
    def train_tree(training_context, training_parameters):
        tree = Tree(training_parameters.maximum_depth)
        rf_operation = RandomForestTrainer.RandomForestOperation(training_context, training_parameters)
        i_start = 0
        i_end = training_context.num_of_samples()
        rf_operation.train_recursive(tree.root, i_start, i_end)
        return tree
