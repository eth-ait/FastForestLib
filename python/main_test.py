from __future__ import division

import numpy as np
import scipy.io
import imp
import os

from tree import level_order_traverse
from forest_trainer import RandomForestTrainer, TrainingParameters
from image_data import ImageDataReader

import c_image_weak_learner as image_weak_learner

def write_config_file(config_file_name, num_of_samples_per_image, maximum_depth, num_of_trees):

    base_config_file_name = "configuration.py"

    f = open(base_config_file_name,'r')
    filedata = f.read()
    f.close()

    sqote="'"
    newdata = filedata.replace(sqote+'num_of_samples_per_image'+sqote + ": 150", sqote+'num_of_samples_per_image'+sqote + " : " + str(num_of_samples_per_image))
    newdata = newdata.replace('maximum_depth=12','maximum_depth=' + str(maximum_depth))
    newdata = newdata.replace('num_of_trees=3','num_of_trees=' + str(num_of_trees))
    newdata = newdata.replace('num_of_thresholds=200','num_of_thresholds=5')
    newdata = newdata.replace('threshold_range_low=-300.0','threshold_range_low=-0.5')
    newdata = newdata.replace('threshold_range_high=+300.0','threshold_range_high=+0.5')

    f = open(config_file_name,'w')
    f.write(newdata)
    f.close()

def train_forest(matlab_file, forest_file, config, profiler=None):
    # training_parameters = TrainingParameters(maximum_depth=15, num_of_features=50, num_of_thresholds=50,
    #                                          num_of_trees=1, minimum_num_of_samples=100)
    if 'num_of_samples_per_image' not in config.training_data_parameters \
            or config.training_data_parameters['num_of_samples_per_image'] <= 0:
        training_data = ImageDataReader.read_from_matlab_file_with_all_samples(matlab_file,
                                                                            **config.training_data_parameters)
    else:
        training_data = ImageDataReader.read_from_matlab_file_with_random_samples(matlab_file,
                                                                               **config.training_data_parameters)
    sample_indices = training_data.create_sample_indices()
    weak_learner_context = image_weak_learner.WeakLearnerContext(config.weak_learner_parameters, training_data)

    print("Training forest with {} samples ...".format(len(sample_indices)))
    trainer = RandomForestTrainer()
    from time import time
    start_time = time()
    if profiler is not None:
        profiler.enable()
    forest = trainer.train_forest(sample_indices, weak_learner_context, config.training_parameters)
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
                # offset_x1 = offset1 // training_data.image_height
                # offset_y1 = offset1 % training_data.image_height
                # offset_x2 = offset2 // training_data.image_height
                # offset_y2 = offset2 % training_data.image_height
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
    m_dict = dict(config.training_parameters.__dict__)
    # m_dict['forest'] = forest_array
    m_dict['forest'] = forest_array
    scipy.io.savemat(forest_file, m_dict)
    print("Done.")

def test_forest(test_data_file, forest_file, config, prediction_output_file=None, profiler=None):
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

    print("Average of normalized confusion matrix diagonal:")
    print np.average(norm_confusion_matrix.diagonal())

    per_pixel_acc = np.average(norm_confusion_matrix.diagonal())

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

    print("Average of normalized confusion matrix diagonal:")
    print np.average(norm_confusion_matrix.diagonal())

    per_frame_acc = np.average(norm_confusion_matrix.diagonal())

    if profiler is not None:
        profiler.disable()
    stop_time = time()
    print("Testing time: {}".format(stop_time - start_time))

    if prediction_output_file is not None:
        print("Writing output file ...")
        scipy.io.savemat(prediction_output_file, output_mat_dict)
        print("Done.")

    return (per_pixel_acc, per_frame_acc)

def load_configuration_from_python_file(filename):
    return imp.load_source('configuration', filename)

def print_forest_prams(config):
    print("")
    print("**FOREST CONFIGURATION**")
    print("#Samples: " + str(config.training_data_parameters['num_of_samples_per_image']))
    print("#Trees: " + str(config.training_parameters.numOfTrees))
    print("Tree Depth: " + str(config.training_parameters.maximumDepth))
    print("")

if __name__ == '__main__':

    base_directory = "results/"
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    #specify which kind of data you will be using for this test
    image_size_x = 64;
    image_size_y = 64;
    num_gestures = 7;
    decimation_factor = 1;
    train_data_file_name = "../data/perturbated_" + str(num_gestures) + \
                           "_gestures_train_decimated_factor_" + str(decimation_factor) + \
                           "_" + str(image_size_x) + "_" + str(image_size_y) + ".mat"
    test_data_file_name = "../data/perturbated_" + str(num_gestures) + \
                          "_gestures_test_decimated_factor_" + str(decimation_factor) + \
                          "_" + str(image_size_x) + "_" + str(image_size_y) + ".mat"

    #specify the parameters to train the forest
    the_sampling_rate = [10, 25, 50, 75, 100]
    the_tree_depth = [18, 20, 22]
    the_forest_size = [1, 2, 3, 4, 5]

    #here you'll save the accuracy performances of each configuration
    file = open(base_directory + "finetuning.txt", "w")
    file.write("Tree name\t Per-pixel Acc\t Per-frame acc\n")
    file.flush()
    os.fsync(file.fileno())

    for _sampling in the_sampling_rate:
        for _forest_size in the_forest_size:
            for _depth in the_tree_depth:

                num_of_samples_per_image = round(_sampling * ( (image_size_x*image_size_y)/100.0)); #use _sampling% of total image pixel as default value
                maximum_depth = _depth;
                num_of_trees = _forest_size;

                config_file_name = base_directory + "configuration_" + str(num_gestures) + "_gestures_" + \
                                   str(image_size_x) + "_" + str(image_size_y) + \
                                   "_S" +str(num_of_samples_per_image) + \
                                   "_D" + str(maximum_depth) + \
                                   "_T" + str(num_of_trees) + ".py"

                forest_file_name = base_directory + "forest_" + str(num_gestures) + "_gestures_" + \
                                   str(image_size_x) + "_" + str(image_size_y) + \
                                   "_S" +str(num_of_samples_per_image) + \
                                   "_D" + str(maximum_depth) + \
                                   "_T" + str(num_of_trees) + ".mat"

                prediction_output_file_name = base_directory + "perfomances_" + str(num_gestures) + "_gestures_" + \
                                              str(image_size_x) + "_" + str(image_size_y) + \
                                              "_S" +str(num_of_samples_per_image) + \
                                              "_D" + str(maximum_depth) + \
                                              "_T" + str(num_of_trees) + ".mat"

                print("Test data File: " + test_data_file_name)
                print("Forest File: " + forest_file_name)
                print("Config File: " + config_file_name)

                #create the config file you will be using for this test
                write_config_file(config_file_name, num_of_samples_per_image, maximum_depth, num_of_trees);
                config = load_configuration_from_python_file(config_file_name)
                print_forest_prams(config)
                #run the training function
                print("Training...")
                train_forest(train_data_file_name, forest_file_name, config)
                print("Done!")
                #run the testing function
                print("Testing...")
                per_pixel_acc, per_frame_acc = test_forest(test_data_file_name, forest_file_name, config, prediction_output_file_name)
                print("Done!")
                file.write(forest_file_name + "\t" + str(per_pixel_acc) + "\t" + str(per_frame_acc) + "\n")
                file.flush()
                os.fsync(file.fileno())
    file.close()

    #part1: effect of training size

    #part2: fine-tuning

    #part3: cross validation