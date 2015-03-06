from __future__ import division

import numpy as np

from tree import level_order_traverse
from forest_trainer import RandomForestTrainer, TrainingParameters
from image_data import ImageDataReader
# from image_training_context import ImageDataReader, SparseImageTrainingContext
import c_image_weak_learner as image_weak_learner


def run(forest_file, test_data_file, profiler=None):
    predictor = image_weak_learner.Predictor.read_from_matlab_file(forest_file)
    test_data = ImageDataReader.read_from_matlab_file_with_all_samples(test_data_file)

    from time import time
    start_time = time()
    if profiler is not None:
        profiler.enable()

    print("Computing confusion matrix...")
    confusion_matrix = np.zeros((test_data.num_of_labels, test_data.num_of_labels), dtype=np.int64)
    for i in xrange(test_data.num_of_images):
        print("Testing image {}".format(i + 1))
        image = test_data.data[i, :, :]
        labels = test_data.labels[i, :, :]
        # TODO: labels are switched in this test data set
        # flat_labels = (-labels.reshape((labels.size,))) + 1
        flat_labels = labels.reshape((labels.size,))
        sample_indices = np.arange(image.size, dtype=np.int64)
        sample_indices = sample_indices[flat_labels >= 0]
        aggregate_statistics = predictor.predict_image_aggregate_statistics(sample_indices, image)
        predicted_labels = np.argmax(aggregate_statistics.histogram, 1)
        num_of_matches = np.sum(flat_labels[sample_indices] == predicted_labels)
        num_of_mismatches = sample_indices.size - num_of_matches
        gt_label = np.max(flat_labels)
        for label in xrange(test_data.num_of_labels):
            confusion_matrix[gt_label, label] += np.sum(predicted_labels == label)
        # confusion_matrix[gt_label_index, gt_label_index] += num_of_matches
        # other_label_index = 1 if gt_label_index == 0 else 0
        # confusion_matrix[gt_label_index, other_label_index ] += predicted_labels.size - num_of_matches

    print confusion_matrix

    # normalize confusion matrix
    normalization_coeff = np.asarray(np.sum(confusion_matrix, 1), dtype=np.float64)[:, np.newaxis]
    norm_confusion_matrix = confusion_matrix / normalization_coeff

    print("Normalized confusion matrix:")
    print norm_confusion_matrix

    # def traverse_tree(node, sample_index):
    #     if node.left_child is None or node.left_child.statistics is None or node.right_child.statistics is None:
    #         return node
    #     else:
    #         feature = node.split_point.feature
    #         threshold = node.split_point.threshold
    #         v = training_context.compute_feature_value(sample_index, feature)
    #         if v < threshold:
    #             return traverse_tree(node.left_child, sample_index)
    #         else:
    #             return traverse_tree(node.right_child, sample_index)
    #
    # predicted_labels = np.empty(sample_indices.shape, dtype=np.int)
    # for i, sample_index in enumerate(sample_indices):
    #     average_histogram = np.zeros((2,), dtype=np.int)
    #     for tree in forest:
    #         leaf_node = traverse_tree(tree.root, sample_index)
    #         average_histogram += leaf_node.statistics.histogram
    #     predicted_label = np.argmax(average_histogram)
    #     predicted_labels[i] = predicted_label
    #
    # print("Matches: {}".format(np.sum(image_data.flat_labels[sample_indices] == predicted_labels)))
    # print("Accuracy: {}".format(np.sum(image_data.flat_labels[sample_indices] == predicted_labels) / len(sample_indices)))

    if profiler is not None:
        profiler.disable()
    stop_time = time()
    print("Testing time: {}".format(stop_time - start_time))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python {} <forest file> <test data file>".format(sys.argv[0]))
        sys.exit(1)

    forest_file = sys.argv[1]
    test_data_file = sys.argv[2]

    run(forest_file, test_data_file)
