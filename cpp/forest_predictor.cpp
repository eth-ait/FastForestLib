//
//  forest_predictor.cpp
//  DistRandomForest
//
//  Created by Benjamin Hepp on 04/02/16.
//
//

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <map>

#include <boost/filesystem/path.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "logger.h"
#include "depth_forest_trainer.h"
#include "image_weak_learner.h"
#include "evaluation_utils.h"
#include "common.h"

using PixelT = ait::CommonPixelT;
using ImageProviderT = ait::CommonImageProviderT;
using ImageProviderPtrT = ait::CommonImageProviderPtrT;
using ImageT = ait::CommonImageT;
using ImagePtrT = ait::CommonImagePtrT;

using SampleT = ait::ImageSample<PixelT>;
using StatisticsT = ait::HistogramStatistics;
using SplitPointT = ait::ImageSplitPoint<PixelT>;
using RandomEngineT = std::mt19937_64;

using ForestT = ait::Forest<SplitPointT, StatisticsT>;
using SampleProviderT = ait::ImageSampleProvider<RandomEngineT, PixelT>;
using ParametersT = typename SampleProviderT::ParametersT;
using SampleIteratorT = typename SampleProviderT::SampleIteratorT;


int main(int argc, const char* argv[]) {
    try {
        // Parse command line arguments.
        TCLAP::CmdLine cmd("Random forest predictor", ' ', "0.3");
        TCLAP::SwitchArg not_all_samples_arg("s", "not-all-samples", "Do not use all samples for testing (i.e. do not overwrite bagging fraction to 1.0)", cmd, false);
        TCLAP::SwitchArg verbose_arg("v", "verbose", "Be verbose and perform some additional sanity checks", cmd, false);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file of the forest to load", false, "forest.json", "string");
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file of the forest to load", false, "forest.bin", "string");
        TCLAP::ValueArg<int> background_label_arg("l", "background-label", "Lower bound of background labels to be ignored", false, -1, "int", cmd);
        TCLAP::ValueArg<std::string> config_file_arg("c", "config", "YAML file with training parameters", false, "", "string", cmd);
        cmd.xorAdd(json_forest_file_arg, binary_forest_file_arg);
        TCLAP::ValueArg<int> num_of_classes_arg("n", "num-of-classes", "Number of classes in the data", false, 1, "int", cmd);
#if WITH_MATLAB
        TCLAP::ValueArg<std::string> data_mat_file_arg("d", "data-file", "File containing image data", false, "", "string");
        TCLAP::ValueArg<std::string> image_list_file_arg("i", "image-list-file", "File containing the names of image files", false, "", "string");
        cmd.xorAdd(data_mat_file_arg, image_list_file_arg);
#else
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", true, "", "string", cmd);
#endif
        cmd.parse(argc, argv);
        
        // Initialize parameters to defaults or load from file.
        ParametersT parameters;
        if (config_file_arg.isSet()) {
            ait::log_info(false) << "Reading config file " << config_file_arg.getValue() << "... " << std::flush;
			rapidjson::Document config_doc;
			ait::ConfigurationUtils::read_configuration_file(config_file_arg.getValue(), config_doc);
            if (config_doc.HasMember("testing_parameters")) {
                parameters.read_from_config(config_doc["testing_parameters"]);
            }
            ait::log_info(false) << " Done." << std::endl;
        }
        // Modify parameters to retrieve all pixels per sample. This can be overwritten by config file.
        if (!not_all_samples_arg.getValue()) {
        	parameters.samples_per_image_fraction = 1.0;
        }

        ForestT forest;
        if (json_forest_file_arg.isSet()) {
            // Read forest from JSON file.
            ait::read_forest_from_json_file(json_forest_file_arg.getValue(), forest);
        } else if (binary_forest_file_arg.isSet()) {
            // Read forest from binary file.
        	ait::read_forest_from_binary_file(binary_forest_file_arg.getValue(), forest);
        } else {
            throw("This should never happen. Either a JSON or a binary forest file have to be specified!");
        }


#if AIT_TESTING
        RandomEngineT rnd_engine(11);
#else
        std::random_device rnd_device;
        ait::log_info() << "rnd(): " << rnd_device();
        RandomEngineT rnd_engine(rnd_device());
#endif

        // Create sample provider.
        ImageProviderPtrT image_provider_ptr;
#if WITH_MATLAB
        if (data_mat_file_arg.isSet()) {
            image_provider_ptr = ait::get_image_provider_from_matlab_file(data_mat_file_arg.getValue());
        } else {
            const std::string image_list_file = image_list_file_arg.getValue();
            image_provider_ptr = ait::get_image_provider_from_image_list(image_list_file_arg.getValue());
        }
#else
        const std::string image_list_file = image_list_file_arg.getValue();
        image_provider_ptr = ait::get_image_provider_from_image_list(image_list_file_arg.getValue());
#endif

        // Retrieve number of classes
        int num_of_classes;
        if (num_of_classes_arg.isSet()) {
        	num_of_classes = num_of_classes_arg.getValue();
        } else {
            ait::log_info() << "Computing number of classes ...";
        	num_of_classes = ait::compute_num_of_classes(image_provider_ptr);
            ait::log_info() << " Found " << num_of_classes << " classes.";
        }

        // Set lower bound for background pixel lables
        ait::label_type background_label;
        if (background_label_arg.isSet()) {
            background_label = background_label_arg.getValue();
        } else {
#if WITH_MATLAB
        	if (data_mat_file_arg.isSet()) {
        		background_label = -1;
        	} else {
        		background_label = num_of_classes;
        	}
#else
			background_label = num_of_classes;
#endif
        }
        parameters.background_label = background_label;

        if (verbose_arg.getValue()) {
        	ait::print_image_size(image_provider_ptr);
        	if (!ait::validate_data_ranges(image_provider_ptr, num_of_classes, parameters.background_label)) {
        		throw std::runtime_error("Foreground label ranges do not match number of classes: " + std::to_string(num_of_classes));
        	}
        }

        // Create sample provider.
        auto sample_provider_ptr = std::make_shared<SampleProviderT>(image_provider_ptr, parameters);

        ait::log_info(false) << "Creating samples for testing ... " << std::flush;
        ait::load_samples_from_all_images(sample_provider_ptr, rnd_engine);
        ait::log_info(false) << " Done." << std::endl;

    	ait::print_sample_counts(forest, sample_provider_ptr, num_of_classes);
    	ait::print_match_counts(forest, sample_provider_ptr);

//        // Compute single-tree confusion matrix.
//        auto tree_utils = ait::make_tree_utils(*forest.begin());
//        auto single_tree_confusion_matrix = tree_utils.compute_confusion_matrix(samples_start, samples_end);
//        ait::log_info() << "Single-tree confusion matrix:" << std::endl << single_tree_confusion_matrix;
//        auto single_tree_norm_confusion_matrix = ait::EvaluationUtils::normalize_confusion_matrix(single_tree_confusion_matrix);
//        ait::log_info() << "Single-tree normalized confusion matrix:" << std::endl << single_tree_norm_confusion_matrix;
//        ait::log_info() << "Single-tree diagonal of normalized confusion matrix:" << std::endl << single_tree_norm_confusion_matrix.diagonal();

        ait::log_info() << "Computing per-pixel confusion matrix.";
    	ait::print_per_pixel_confusion_matrix(forest, sample_provider_ptr, num_of_classes);

//        using ConfusionMatrixType = typename decltype(tree_utils)::MatrixType;
//        // Computing single-tree per-frame confusion matrix
//        ait::log_info() << "Computing per-frame confusion matrix.";
//        ConfusionMatrixType per_frame_single_tree_confusion_matrix(num_of_classes, num_of_classes);
//        per_frame_single_tree_confusion_matrix.setZero();
//        // TODO: This should be configurable by input file
//        SampleProviderT sample_provider(image_list, full_parameters);
//        for (int i = 0; i < image_list.size(); ++i)
//        {
//            sample_provider.clear_samples();
//            sample_provider.load_samples_from_image(i, rnd_engine);
//            samples_start = sample_provider.get_samples_begin();
//            samples_end = sample_provider.get_samples_end();
//            tree_utils.update_confusion_matrix(per_frame_single_tree_confusion_matrix, samples_start, samples_end);
//        }
//        ait::log_info() << "Single-tree per-frame confusion matrix:" << std::endl << per_frame_single_tree_confusion_matrix;
//        auto per_frame_single_tree_norm_confusion_matrix = ait::EvaluationUtils::normalize_confusion_matrix(per_frame_single_tree_confusion_matrix);
//        ait::log_info() << "Single-tree normalized per-frame confusion matrix:" << std::endl << per_frame_single_tree_norm_confusion_matrix;
//        ait::log_info() << "Single-tree diagonal of normalized per-frame confusion matrix:" << std::endl << per_frame_single_tree_norm_confusion_matrix.diagonal();
//        ait::log_info() << "Single-tree mean of diagonal of normalized per-frame confusion matrix:" << std::endl << per_frame_single_tree_norm_confusion_matrix.diagonal().mean();

        ait::log_info() << "Computing per-frame confusion matrix.";
    	ait::print_per_frame_confusion_matrix(forest, sample_provider_ptr, rnd_engine, num_of_classes);

    } catch (const TCLAP::ArgException &e) {
        ait::log_error() << "Error parsing command line: " << e.error() << " for arg " << e.argId();
    }
    
    return 0;
}
