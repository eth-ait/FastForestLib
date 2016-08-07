//
//  depth_trainer.cpp
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include <memory>

#include <tclap/CmdLine.h>

#include "ait.h"
#include "depth_forest_trainer.h"
#include "histogram_statistics.h"
#include "image_weak_learner.h"
#include "bagging_wrapper.h"
#include "common.h"


using PixelT = ait::CommonPixelT;
using ImageProviderT = ait::CommonImageProviderT;
using ImageProviderPtrT = ait::CommonImageProviderPtrT;
using ImageT = ait::CommonImageT;
using ImagePtrT = ait::CommonImagePtrT;

using SampleT = ait::ImageSample<PixelT>;
using StatisticsT = ait::HistogramStatistics;
using RandomEngineT = std::mt19937_64;

using SampleProviderT = ait::ImageSampleProvider<RandomEngineT, PixelT>;

template <class TSampleIterator> using WeakLearnerAliasT
    = typename ait::ImageWeakLearner<StatisticsT::Factory, TSampleIterator, RandomEngineT>;

template <class TSampleIterator> using ForestTrainerAliasT
    = ait::DepthForestTrainer<WeakLearnerAliasT, TSampleIterator>;

using BaggingWrapperT = ait::BaggingWrapper<ForestTrainerAliasT, SampleProviderT>;
using SampleIteratorT = BaggingWrapperT::SampleIteratorT;
using ForestTrainerT = BaggingWrapperT::ForestTrainerT;
using WeakLearnerT = ForestTrainerT::WeakLearnerT;


int main(int argc, const char* argv[]) {
    try {
    	AIT_LOG_DEBUG("Running in debug mode");

        // Parse command line arguments.
        TCLAP::CmdLine cmd("Depth RF trainer", ' ', "0.3");
        TCLAP::SwitchArg verbose_arg("v", "verbose", "Be verbose and perform some additional sanity checks", cmd, false);
        TCLAP::SwitchArg print_confusion_matrix_switch("m", "conf-matrix", "Print confusion matrix", cmd, true);
        TCLAP::ValueArg<int> background_label_arg("l", "background-label", "Lower bound of background labels to be ignored", false, -1, "int", cmd);
        TCLAP::ValueArg<std::string> config_file_arg("c", "config", "YAML file with training parameters", false, "", "string", cmd);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string");
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string");
        cmd.xorAdd(json_forest_file_arg, binary_forest_file_arg);
        TCLAP::ValueArg<int> num_of_classes_arg("n", "num-of-classes", "Number of classes in the data", false, 1, "int", cmd);
#if AIT_MULTI_THREADING
        TCLAP::ValueArg<int> num_of_threads_arg("t", "threads", "Number of threads to use", false, -1, "int", cmd);
#endif
#if WITH_MATLAB
        TCLAP::ValueArg<std::string> data_mat_file_arg("d", "data-file", "File containing image data", false, "", "string");
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", false, "", "string");
        cmd.xorAdd(data_mat_file_arg, image_list_file_arg);
#else
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", true, "", "string", cmd);
#endif
        cmd.parse(argc, argv);

        bool print_confusion_matrix = print_confusion_matrix_switch.getValue();

        // Initialize training and weak-learner parameters to defaults or load from file
        ForestTrainerT::ParametersT training_parameters;
        WeakLearnerT::ParametersT weak_learner_parameters;
        if (config_file_arg.isSet()) {
            ait::log_info(false) << "Reading config file " << config_file_arg.getValue() << "... " << std::flush;
			rapidjson::Document config_doc;
			ait::ConfigurationUtils::read_configuration_file(config_file_arg.getValue(), config_doc);
            if (config_doc.HasMember("training_parameters")) {
                training_parameters.read_from_config(config_doc["training_parameters"]);
            }
            if (config_doc.HasMember("weak_learner_parameters")) {
                weak_learner_parameters.read_from_config(config_doc["weak_learner_parameters"]);
            }
            ait::log_info(false) << " Done." << std::endl;
        }
#if AIT_MULTI_THREADING
        if (num_of_threads_arg.isSet()) {
            training_parameters.num_of_threads = num_of_threads_arg.getValue();
        }
#endif

        // Create image provider.
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
            ait::log_info(false) << "Computing number of classes ..." << std::flush;
        	num_of_classes = ait::compute_num_of_classes(image_provider_ptr);
            ait::log_info(false) << " Found " << num_of_classes << " classes." << std::endl;
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
        weak_learner_parameters.background_label = background_label;

        if (verbose_arg.getValue()) {
        	ait::print_image_size(image_provider_ptr);
        	if (!ait::validate_data_ranges(image_provider_ptr, num_of_classes, weak_learner_parameters.background_label)) {
        		throw std::runtime_error("Foreground label ranges do not match number of classes: " + num_of_classes);
        	}
        }

        // Create sample provider.
        auto sample_provider_ptr = std::make_shared<SampleProviderT>(image_provider_ptr, weak_learner_parameters);

        // Create weak learner and trainer.
        StatisticsT::Factory statistics_factory(num_of_classes);
        WeakLearnerT iwl(weak_learner_parameters, statistics_factory);
        ForestTrainerT trainer(iwl, training_parameters);
        BaggingWrapperT bagging_wrapper(trainer, sample_provider_ptr);

#ifdef AIT_TESTING
        RandomEngineT rnd_engine(11);
#else
        std::random_device rnd_device;
        ait::log_info() << "rnd(): " << rnd_device();
        RandomEngineT rnd_engine(rnd_device());
#endif

        // Train a forest and time it.
        ait::log_info() << "Starting training with " << training_parameters.num_of_threads << " threads ...";
        auto start_time = std::chrono::high_resolution_clock::now();
        ForestTrainerT::ForestT forest = bagging_wrapper.train_forest(rnd_engine);
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = stop_time - start_time;
        auto period = std::chrono::high_resolution_clock::period();
        double elapsed_seconds = duration.count() * period.num / static_cast<double>(period.den);
        ait::log_info() << "Done.";
        ait::log_info() << "Running time: " << elapsed_seconds;
        
        // Optionally: Serialize forest to JSON file.
        if (json_forest_file_arg.isSet()) {
            ait::write_forest_to_json_file(json_forest_file_arg.getValue(), forest);
        // Optionally: Serialize forest to binary file.
        } else if (binary_forest_file_arg.isSet()) {
            ait::write_forest_to_binary_file(binary_forest_file_arg.getValue(), forest);
        } else {
            throw("This should never happen. Either a JSON or a binary forest file have to be specified!");
        }

        // Optionally: Compute some stats and print them.
        if (print_confusion_matrix) {
            ait::log_info(false) << "Creating samples for testing ... " << std::flush;
            ait::load_samples_from_all_images(sample_provider_ptr, rnd_engine);
            ait::log_info(false) << " Done." << std::endl;

        	ait::print_sample_counts(forest, sample_provider_ptr);
        	ait::print_match_counts(forest, sample_provider_ptr);

            ait::log_info() << "Computing per-pixel confusion matrix.";
        	ait::print_per_pixel_confusion_matrix(forest, sample_provider_ptr);

            ait::log_info() << "Computing per-frame confusion matrix.";
            WeakLearnerT::ParametersT full_parameters(weak_learner_parameters);
            // Modify parameters to retrieve all pixels per sample
            full_parameters.samples_per_image_fraction = 1.0;
            // Create sample provider.
            auto full_sample_provider_ptr = std::make_shared<SampleProviderT>(image_provider_ptr, weak_learner_parameters);
        	ait::print_per_frame_confusion_matrix(forest, full_sample_provider_ptr, rnd_engine, num_of_classes);
        }
    } catch (const TCLAP::ArgException& e) {
        std::cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    } catch (const std::runtime_error& error) {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}

