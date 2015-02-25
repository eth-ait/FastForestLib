from __future__ import division

import numpy as np

from forest_trainer import RandomForestTrainer, TrainingParameters
# from image_training_context import ImageDataReader, SparseImageTrainingContext
from c_image_training_context import ImageDataReader, SparseImageTrainingContext


if __name__ == '__main__':

    import sys
    if len(sys.argv) < 3:
        print("Usage: python {} <matlab file> <number of samples per image>".format(sys.argv[0]))
        sys.exit(1)

    matlab_file = sys.argv[1]
    num_of_samples_per_image = int(sys.argv[2])

    training_parameters = TrainingParameters(maximum_depth=10, num_of_features=50, num_of_thresholds=50)
    data = ImageDataReader.read_from_matlab_file(matlab_file, num_of_samples_per_image)
    sample_indices = data.create_sample_indices()
    training_context = SparseImageTrainingContext(data)

    trainer = RandomForestTrainer()
    forest = trainer.train_forest(sample_indices, training_context, training_parameters)

    def traverse_tree(node, sample_index):
        if node.left_child is None or node.left_child.statistics is None or node.right_child.statistics is None:
            return node
        else:
            feature = node.split_point.feature
            threshold = node.split_point.threshold
            v = training_context.compute_feature_value(sample_index, feature)
            if v < threshold:
                return traverse_tree(node.left_child, sample_index)
            else:
                return traverse_tree(node.right_child, sample_index)

    predicted_labels = np.empty(sample_indices.shape, dtype=np.int)
    for i, sample_index in enumerate(sample_indices):
        average_histogram = np.zeros((2,), dtype=np.int)
        for tree in forest:
            leaf_node = traverse_tree(tree.root, sample_index)
            average_histogram += leaf_node.statistics.histogram
        predicted_label = np.argmax(average_histogram)
        predicted_labels[i] = predicted_label

    print("Accuracy: {}".format(np.sum(data.flat_labels[sample_indices] == predicted_labels) / len(sample_indices)))
