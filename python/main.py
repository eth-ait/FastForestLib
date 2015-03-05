from __future__ import division

import numpy as np

from tree import level_order_traverse
from forest_trainer import RandomForestTrainer, TrainingParameters
from image_data import ImageDataReader
# from image_training_context import ImageDataReader, SparseImageTrainingContext
import c_image_training_context as image_training_context


def run(matlab_file, num_of_samples_per_image, forest_file, profiler=None):
    # training_parameters = TrainingParameters(maximum_depth=15, num_of_features=50, num_of_thresholds=50,
    #                                          num_of_trees=1, minimum_num_of_samples=100)
    training_parameters = TrainingParameters(maximum_depth=10, num_of_features=10, num_of_thresholds=10,
                                             num_of_trees=1, minimum_num_of_samples=100)
    if num_of_samples_per_image < 0:
        image_data = ImageDataReader.read_from_matlab_file_with_all_samples(matlab_file)
    else:
        image_data = ImageDataReader.read_from_matlab_file_with_random_samples(matlab_file, num_of_samples_per_image)
    sample_indices = image_data.create_sample_indices()
    feature_offset_range_low = 3
    feature_offset_range_high = 15
    threshold_range_low = -300.
    threshold_range_high = +300.
    training_context_parameters = image_training_context.Parameters(feature_offset_range_low, feature_offset_range_high, threshold_range_low, threshold_range_high)
    training_context = image_training_context.SparseImageTrainingContext(training_context_parameters, image_data)

    trainer = RandomForestTrainer()
    from time import time
    start_time = time()
    if profiler is not None:
        profiler.enable()
    forest = trainer.train_forest(sample_indices, training_context, training_parameters)
    if profiler is not None:
        profiler.disable()
    stop_time = time()
    print("Training time: {}".format(stop_time - start_time))

    def convert_tree_to_matrix(tree):
        split_point1 = tree.root.split_point
        feature_len = len(split_point1.feature)
        statistics1 = tree.root.statistics
        num_of_labels = statistics1.histogram.shape[0]
        num_of_columns = feature_len + 1 + num_of_labels + 1
        matrix = -np.ones((len(tree), num_of_columns), dtype=np.float)
        assert(isinstance(matrix, np.ndarray))
        for i, node in enumerate(level_order_traverse(tree.root)):
            if node.statistics is None:
                continue
            if not node.leaf_node:
                split_point = node.split_point
                feature = split_point.feature
                threshold = split_point.threshold
                # offset1, offset2 = feature
                # offset_x1 = offset1 // image_data.image_height
                # offset_y1 = offset1 % image_data.image_height
                # offset_x2 = offset2 // image_data.image_height
                # offset_y2 = offset2 % image_data.image_height
                offset_x1, offset_y1, offset_x2, offset_y2 = feature
                matrix[i, :4] = (offset_x1, offset_y1, offset_x2, offset_y2)
                matrix[i, 4] = threshold
            histogram = node.statistics.histogram
            matrix[i, 5:5+num_of_labels] = histogram
            leaf_node_indicator = 1 if node.leaf_node else 0
            matrix[i, -1] = leaf_node_indicator
        return matrix

    print("Saving tree...")
    forest_array = np.zeros((len(forest),), dtype=np.object)
    for i, tree in enumerate(forest):
        forest_array[i] = convert_tree_to_matrix(tree)
    import scipy.io
    # m_dict = dict(training_parameters)
    m_dict = dict(training_parameters.__dict__)
    # m_dict['forest'] = forest_array
    m_dict['forest'] = forest_array
    scipy.io.savemat(forest_file, m_dict)
    print("Done.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python {} <matlab file> <number of samples per image> <forest file>".format(sys.argv[0]))
        sys.exit(1)

    matlab_file = sys.argv[1]
    num_of_samples_per_image = int(sys.argv[2])
    forest_file = sys.argv[3]

    run(matlab_file, num_of_samples_per_image, forest_file)

