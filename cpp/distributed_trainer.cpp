//
//  distributed_trainer.cpp
//  DistRandomForest
//
//  Created by Benjamin Hepp on 30/09/15.
//
//

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "distributed_forest_trainer.h"
#include "distributed_bagging_wrapper.h"
#include "image_weak_learner.h"
#include "common.h"


using PixelT = ait::CommonPixelT;
using ImageProviderT = ait::CommonImageProviderT;
using ImageProviderPtrT = ait::CommonImageProviderPtrT;
using ImageT = ait::CommonImageT;
using ImagePtrT = ait::CommonImagePtrT;

using SampleT = ait::ImageSample<PixelT>;
using StatisticsT = ait::HistogramStatistics;
using RandomEngineT = std::mt19937_64;

using SampleProviderT = ait::ImageSampleProvider<RandomEngineT>;

template <typename TSampleIterator> using WeakLearnerAliasT
    = typename ait::ImageWeakLearner<StatisticsT::Factory, TSampleIterator, RandomEngineT, PixelT>;

template <class TSampleIterator> using ForestTrainerAliasT
    = typename ait::DistributedForestTrainer<WeakLearnerAliasT, TSampleIterator>;

using BaggingWrapperT = ait::DistributedBaggingWrapper<ForestTrainerAliasT, SampleProviderT>;
using SampleIteratorT = typename BaggingWrapperT::SampleIteratorT;
using ForestTrainerT = typename BaggingWrapperT::ForestTrainerT;
using WeakLearnerT = typename ForestTrainerT::WeakLearnerT;

int main(int argc, const char* argv[]) {
    try {
        // Initialize MPI.
        boost::mpi::environment env;
        boost::mpi::communicator world;
        std::ostringstream prefix_stream;
        prefix_stream << world.rank() << "> ";
        ait::logger().set_prefix(prefix_stream.str());
        ait::log_info() << "Rank " << world.rank() << " of " << world.size() << ".";

        MPI::COMM_WORLD.Set_errhandler ( MPI::ERRORS_THROW_EXCEPTIONS );

        // Parse command line arguments.
        TCLAP::CmdLine cmd("Distributed RF trainer", ' ', "0.3");
        TCLAP::SwitchArg verbose_arg("v", "verbose", "Be verbose and perform some additional sanity checks", cmd, false);
        TCLAP::SwitchArg hide_confusion_matrix_switch("m", "no-conf-matrix", "Don't print confusion matrix", cmd, false);
        TCLAP::ValueArg<int> background_label_arg("l", "background-label", "Lower bound of background labels to be ignored", false, -1, "int", cmd);
#if AIT_MULTI_THREADING
        TCLAP::ValueArg<int> num_of_threads_arg("t", "threads", "Number of threads to use", false, 1, "int", cmd);
#endif
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string");
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string");
        cmd.xorAdd(json_forest_file_arg, binary_forest_file_arg);
        TCLAP::ValueArg<std::string> config_file_arg("c", "config", "YAML file with training parameters", false, "", "string", cmd);
#if WITH_MATLAB
        TCLAP::ValueArg<std::string> data_mat_file_arg("d", "data-file", "File containing image data", false, "", "string");
        TCLAP::ValueArg<std::string> image_list_file_arg("i", "image-list-file", "File containing the names of image files", false, "", "string");
        TCLAP::ValueArg<int> num_of_classes_arg("n", "num-of-classes", "Number of classes in the data", false, 1, "int", cmd);
        cmd.xorAdd(data_mat_file_arg, image_list_file_arg);
#else
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", true, "", "string", cmd);
        TCLAP::ValueArg<int> num_of_classes_arg("n", "num-of-classes", "Number of classes in the data", true, 1, "int", cmd);
#endif
        cmd.parse(argc, argv);
#if WITH_MATLAB
        if (image_list_file_arg.isSet() && !num_of_classes_arg.isSet()) {
            cmd.getOutput()->usage(cmd);
            throw std::runtime_error("Number of classes needs to be specified when using a list of image files.");
        }
#endif


        const bool print_confusion_matrix = !hide_confusion_matrix_switch.getValue();
        const std::string image_list_file = image_list_file_arg.getValue();
        
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
        	if (ait::validate_data_ranges(image_provider_ptr, num_of_classes, weak_learner_parameters.background_label)) {
        		throw std::runtime_error("Foreground label ranges do not match number of classes: " + num_of_classes);
        	}
        }

        // Create sample provider.
        auto sample_provider_ptr = std::make_shared<SampleProviderT>(image_provider_ptr, weak_learner_parameters);

        // Create weak learner and trainer.
        StatisticsT::Factory statistics_factory(num_of_classes);
        WeakLearnerT iwl(weak_learner_parameters, statistics_factory);
        ForestTrainerT trainer(world, iwl, training_parameters);
        BaggingWrapperT bagging_wrapper(world, trainer, sample_provider_ptr);

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
        if (world.rank() == 0)
        {
            ait::log_info() << "Running time: " << elapsed_seconds;
        }
        
        if (world.rank() == 0)
        {
            // Optionally: Serialize forest to JSON file.
            if (json_forest_file_arg.isSet())
            {
                {
                    ait::log_info(false) << "Writing json forest file " << json_forest_file_arg.getValue() << "... " << std::flush;
                    std::ofstream ofile(json_forest_file_arg.getValue());
                    cereal::JSONOutputArchive oarchive(ofile);
                    oarchive(cereal::make_nvp("forest", forest));
                    ait::log_info(false) << " Done." << std::endl;
                }
            }
            // Optionally: Serialize forest to binary file.
            else if (binary_forest_file_arg.isSet())
            {
                {
                    ait::log_info(false) << "Writing binary forest file " << binary_forest_file_arg.getValue() << "... " << std::flush;
                    std::ofstream ofile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                    cereal::BinaryOutputArchive oarchive(ofile);
                    oarchive(cereal::make_nvp("forest", forest));
                    ait::log_info(false) << " Done." << std::endl;
                }
            }
            else
            {
                throw("This should never happen. Either a JSON or a binary forest file have to be specified!");
            }

            // Optionally: Compute some stats and print them.
            if (print_confusion_matrix) {
                ait::log_info(false) << "Creating samples for testing ... " << std::flush;
                ait::load_samples_from_all_images(sample_provider_ptr, rnd_engine);
                ait::log_info(false) << " Done." << std::endl;

            	ait::print_sample_counts(forest, sample_provider_ptr, num_of_classes);
            	ait::print_match_counts(forest, sample_provider_ptr);

                ait::log_info() << "Computing per-pixel confusion matrix.";
            	ait::print_per_pixel_confusion_matrix(forest, sample_provider_ptr, num_of_classes);

                ait::log_info() << "Computing per-frame confusion matrix.";
                WeakLearnerT::ParametersT full_parameters(weak_learner_parameters);
                // Modify parameters to retrieve all pixels per sample
                full_parameters.samples_per_image_fraction = 1.0;
                // Create sample provider.
                auto full_sample_provider_ptr = std::make_shared<SampleProviderT>(image_provider_ptr, weak_learner_parameters);
            	ait::print_per_frame_confusion_matrix(forest, full_sample_provider_ptr, rnd_engine, num_of_classes);
            }
        }
    }
    catch (const TCLAP::ArgException& e)
    {
        ait::log_error() << "Error parsing command line: " << e.error() << " for arg " << e.argId();
    }

    return 0;
}
