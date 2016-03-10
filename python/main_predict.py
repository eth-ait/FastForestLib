from __future__ import division

import numpy as np
import scipy.io
import imp

from image_data import ImageDataReader
# from image_training_context import ImageDataReader, SparseImageTrainingContext
import c_image_weak_learner as image_weak_learner


def run(forest_file, test_data_file, config, prediction_output_file=None, profiler=None):
    predictor = image_weak_learner.Predictor.read_from_matlab_file(forest_file)
    test_data = ImageDataReader.read_from_matlab_file_with_all_samples(test_data_file, **config.testing_data_parameters)

    from time import time
    start_time = time()
    if profiler is not None:
        profiler.enable()

    if prediction_output_file is not None:
        output_mat_dict = {}

    print("Computing per-pixel confusion matrix...")
    confusion_matrix = np.zeros((test_data.num_of_labels, test_data.num_of_labels), dtype=np.int64)
    if prediction_output_file is not None:
        pixel_predicted_labels = - np.ones((test_data.num_of_images, test_data.image_width * test_data.image_height), dtype=np.int64)
    for i in xrange(test_data.num_of_images):
        # print("Testing image {}".format(i + 1))
        image = test_data.data[i, :, :]
        labels = test_data.labels[i, :, :]
        flat_labels = labels.reshape((labels.size,))
        sample_indices = np.arange(image.size, dtype=np.int64)
        sample_indices = sample_indices[flat_labels >= 0]
        if 'max_evaluation_depth' in config.testing_parameters:
            max_evaluation_depth = config.testing_parameters['max_evaluation_depth']
        else:
            max_evaluation_depth = -1
        aggregate_statistics = predictor.predict_image_aggregate_statistics(sample_indices, image,
                                                                            max_evaluation_depth=max_evaluation_depth)
        predicted_labels = np.argmax(aggregate_statistics.histogram, 1)
        # num_of_matches = np.sum(flat_labels[sample_indices] == predicted_labels)
        # gt_label = np.max(flat_labels)
        # for label in xrange(test_data.num_of_labels):
        #     confusion_matrix[gt_label, label] += np.sum(predicted_labels == label)
        for label1 in xrange(test_data.num_of_labels):
            label1_gt_mask = flat_labels[sample_indices] == label1
            for label2 in xrange(test_data.num_of_labels):
                confusion_matrix[label1, label2] += np.sum(predicted_labels[label1_gt_mask] == label2)
        # num_of_mismatches = sample_indices.size - num_of_matches
        # confusion_matrix[gt_label_index, gt_label_index] += num_of_matches
        # other_label_index = 1 if gt_label_index == 0 else 0
        # confusion_matrix[gt_label_index, other_label_index ] += predicted_labels.size - num_of_matches

        if prediction_output_file is not None:
            pixel_predicted_labels[i, sample_indices] = predicted_labels

    if prediction_output_file is not None:
        pixel_predicted_labels = pixel_predicted_labels.reshape((test_data.num_of_images, test_data.image_width, test_data.image_height))
        output_mat_dict['pixel_predicted_labels'] = pixel_predicted_labels
        output_mat_dict['pixel_confusion_matrix'] = confusion_matrix

    # normalize confusion matrix
    normalization_coeff = np.asarray(np.sum(confusion_matrix, 1), dtype=np.float64)[:, np.newaxis]
    norm_confusion_matrix = confusion_matrix / normalization_coeff

    print("Pixel-counts for each label:")
    print normalization_coeff

    print("Non-normalized confusion matrix:")
    print confusion_matrix

    print("Normalized confusion matrix:")
    print norm_confusion_matrix

    print("Diagonal of normalized confusion matrix:")
    print norm_confusion_matrix.diagonal()

    print("Mean of diagonal of normalized confusion matrix:")
    print np.mean(norm_confusion_matrix.diagonal())


    print("")
    print("Computing per-frame confusion matrix...")

    confusion_matrix = np.zeros((test_data.num_of_labels, test_data.num_of_labels), dtype=np.int64)
    if prediction_output_file is not None:
        frame_predicted_labels = - np.ones((test_data.num_of_images,), dtype=np.int64)
    for i in xrange(test_data.num_of_images):
        # print("Testing image {}".format(i + 1))
        image = test_data.data[i, :, :]
        labels = test_data.labels[i, :, :]
        flat_labels = labels.reshape((labels.size,))
        sample_indices = np.arange(image.size, dtype=np.int64)
        sample_indices = sample_indices[flat_labels >= 0]
        if len(sample_indices) == 0:
            continue
        if 'max_evaluation_depth' in config.testing_parameters:
            max_evaluation_depth = config.testing_parameters['max_evaluation_depth']
        else:
            max_evaluation_depth = -1
        aggregate_statistics = predictor.predict_image_aggregate_statistics(sample_indices, image,
                                                                            max_evaluation_depth=max_evaluation_depth)
        predicted_labels = np.argmax(aggregate_statistics.histogram, 1)
        frame_gt_label = np.max(flat_labels[sample_indices])
        predicted_label_hist = np.bincount(predicted_labels, minlength=test_data.num_of_labels)
        frame_label = np.argmax(predicted_label_hist)
        confusion_matrix[frame_gt_label, frame_label] += 1

        if prediction_output_file is not None:
            frame_predicted_labels[i] = frame_label

    if prediction_output_file is not None:
        output_mat_dict['frame_predicted_labels'] = frame_predicted_labels
        output_mat_dict['frame_confusion_matrix'] = confusion_matrix

    # normalize confusion matrix
    normalization_coeff = np.asarray(np.sum(confusion_matrix, 1), dtype=np.float64)[:, np.newaxis]
    norm_confusion_matrix = confusion_matrix / normalization_coeff

    print("Frame-counts for each label:")
    print normalization_coeff

    print("Non-normalized confusion matrix:")
    print confusion_matrix

    print("Normalized confusion matrix:")
    print norm_confusion_matrix

    print("Diagonal of normalized confusion matrix:")
    print norm_confusion_matrix.diagonal()

    print("Mean of diagonal of normalized confusion matrix:")
    print np.mean(norm_confusion_matrix.diagonal())

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
    print("Testing time (secs): {}".format(stop_time - start_time))
    print("Testing time (min): {}".format( (stop_time - start_time) / 60.0))

    if prediction_output_file is not None:
        print("Writing output file ...")
        scipy.io.savemat(prediction_output_file, output_mat_dict)
        print("Done.")


def load_configuration_from_python_file(filename):
    return imp.load_source('configuration', filename)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python {} <test data file> <forest file> [configuration file] [prediction output file]".format(sys.argv[0]))
        sys.exit(1)

    test_data_file = sys.argv[1]
    forest_file = sys.argv[2]

    config_file = 'configuration.py'
    if len(sys.argv) > 3:
        config_file = sys.argv[3]
    config = load_configuration_from_python_file(config_file)

    prediction_output_file = None
    if len(sys.argv) > 4:
        prediction_output_file = sys.argv[4]

    run(forest_file, test_data_file, config, prediction_output_file)
