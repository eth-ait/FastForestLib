from forest import Forest
from tree import Tree
from training_context import TrainingContext


class TrainingParameters:

    def __init__(self, num_of_trees=3, maximum_depth=20, num_of_features=100,
                 num_of_thresholds=50, minimum_num_of_samples=100):
        self.num_of_trees = num_of_trees
        self.maximum_depth = maximum_depth
        self.num_of_features = num_of_features
        self.num_of_thresholds = num_of_thresholds
        self.minimum_information_gain = 0
        self.minimum_num_of_samples = minimum_num_of_samples


class RandomForestTrainer:

    class _TrainingOperation:

        def __init__(self, sample_indices, training_context, training_parameters):
            assert isinstance(training_context, TrainingContext)
            assert isinstance(training_parameters, TrainingParameters)
            self._sample_indices = sample_indices
            self._trainingContext = training_context
            self._trainingParameters = training_parameters

        def train_recursive(self, node, i_start, i_end, statistics=None):

            # define local aliases for some long variable names
            sample_indices = self._sample_indices[i_start:i_end]

            # assign statistics to node
            if statistics is None:
                statistics = self._trainingContext.compute_statistics(sample_indices)
            node.statistics = statistics

            # stop splitting the node if it is a leaf node
            if node.left_child is None:
                return

            # stop splitting the node if the minimum number of samples has been reached
            if i_end - i_start < self._trainingParameters.minimum_num_of_samples:
                return

            split_point_context = self._trainingContext.sample_split_points(
                sample_indices,
                self._trainingParameters.num_of_features,
                self._trainingParameters.num_of_thresholds)

            # TODO: distribute features and thresholds to ranks > 0

            # compute the statistics for all feature and threshold combinations
            split_point_context.compute_split_statistics()

            # TODO: send statistics to rank 0
            # send split_point_context.get_split_statistics_buffer()

            # TODO: receive statistics from rank > 0
            # for received statistics
            #     split_point_context.accumulate_split_statistics(statistics)

            # find the best feature (only on rank 0)
            best_split_point_id, best_information_gain \
                = split_point_context.select_best_split_point(node.statistics, return_information_gain=True)

            # TODO: send best feature, threshold and information gain to ranks > 0

            # TODO: move criterion into trainingContext?
            # stop splitting the node if the best information gain is below the minimum information gain
            if best_information_gain < self._trainingParameters.minimum_information_gain:
                return

            # partition sample_indices according to the selected feature and threshold.
            # i.e. sample_indices[:i_split] will contain the left child indices
            # and sample_indices[i_split:] will contain the right child indices
            i_split = split_point_context.partition(sample_indices, best_split_point_id)

            node.splitPoint = split_point_context.get_split_point(best_split_point_id)

            # TODO: can we reuse computed statistics from split_point_context???
            left_child_statistics = None
            right_child_statistics = None

            # train left and right child
            self.train_recursive(node.left_child, i_start, i_split, left_child_statistics)
            self.train_recursive(node.right_child, i_split, i_end, right_child_statistics)

    def train_forest(self, sample_indices, training_context, training_parameters):
        forest = Forest()
        for i in xrange(training_parameters.num_of_trees):
            tree = self.train_tree(sample_indices, training_context, training_parameters)
            forest.append(tree)
        return forest

    def train_tree(self, sample_indices, training_context, training_parameters):
        tree = Tree(training_parameters.maximum_depth)
        rf_operation = self._TrainingOperation(sample_indices, training_context, training_parameters)
        i_start = 0
        i_end = len(sample_indices)
        rf_operation.train_recursive(tree.root, i_start, i_end)
        return tree
