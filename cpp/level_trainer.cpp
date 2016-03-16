//
//  level_trainer.cpp
//  DistRandomForest
//
//  Created by Benjamin Hepp on 30/09/15.
//
//

#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>

#include <boost/filesystem/path.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "logger.h"
#include "level_forest_trainer.h"
#include "bagging_wrapper.h"
#include "image_weak_learner.h"
#include "csv_utils.h"
#include "evaluation_utils.h"

using PixelT = ait::pixel_type;
using ImageT = ait::Image<PixelT>;
using SampleT = ait::ImageSample<PixelT>;
using StatisticsT = ait::HistogramStatistics;
using RandomEngineT = std::mt19937_64;

using SampleProviderT = ait::ImageSampleProvider<RandomEngineT>;

template <class TSampleIterator> using WeakLearnerAliasT
    = ait::ImageWeakLearner<StatisticsT::Factory, TSampleIterator, RandomEngineT, PixelT>;

template <class TSampleIterator> using ForestTrainerAliasT
    = ait::LevelForestTrainer<WeakLearnerAliasT, TSampleIterator>;

using BaggingWrapperT = ait::BaggingWrapper<ForestTrainerAliasT, SampleProviderT>;
using SampleIteratorT = BaggingWrapperT::SampleIteratorT;
using ForestTrainerT = BaggingWrapperT::ForestTrainerT;
using WeakLearnerT = ForestTrainerT::WeakLearnerT;

int main(int argc, const char* argv[]) {
    try {
        // Parse command line arguments.
        TCLAP::CmdLine cmd("Level RF trainer", ' ', "0.3");
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", true, "", "string", cmd);
        TCLAP::ValueArg<int> num_of_classes_arg("n", "num-of-classes", "Number of classes in the data", true, 1, "int", cmd);
        TCLAP::SwitchArg hide_confusion_matrix_switch("m", "no-conf-matrix", "Don't print confusion matrix", cmd, false);
        TCLAP::ValueArg<int> background_label_arg("l", "background-label", "Lower bound of background labels to be ignored", false, -1, "int", cmd);
        TCLAP::SwitchArg save_regularly_switch("", "save-regularly", "Whether forest and trees should be saved regularly", cmd, false);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string");
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string");
        TCLAP::ValueArg<std::string> config_file_arg("c", "config", "YAML file with training parameters", false, "", "string", cmd);
#if AIT_MULTI_THREADING
        TCLAP::ValueArg<int> num_of_threads_arg("t", "threads", "Number of threads to use", false, -1, "int", cmd);
#endif
        cmd.xorAdd(json_forest_file_arg, binary_forest_file_arg);
        cmd.parse(argc, argv);

        const int num_of_classes = num_of_classes_arg.getValue();
        const bool save_regularly = save_regularly_switch.getValue();
        const bool print_confusion_matrix = !hide_confusion_matrix_switch.getValue();
        const std::string image_list_file = image_list_file_arg.getValue();
        
        // Initialize training and weak-learner parameters to defaults or load from file
        ForestTrainerT::ParametersT training_parameters;
        WeakLearnerT::ParametersT weak_learner_parameters;
        if (config_file_arg.isSet()) {
            ait::log_info(false) << "Reading config file " << config_file_arg.getValue() << "... " << std::flush;
            rapidjson::Document config_doc = ait::ConfigurationUtils::read_configuration_file(config_file_arg.getValue());
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

        // Read image file list
        ait::log_info(false) << "Reading image list ... " << std::flush;
        std::vector<std::tuple<std::string, std::string>> image_list;
        std::ifstream ifile(image_list_file);
        if (!ifile.good())
        {
            throw std::runtime_error("Unable to open image list file");
        }
        ait::CSVReader<std::string> csv_reader(ifile);
        for (auto it = csv_reader.begin(); it != csv_reader.end(); ++it)
        {
            if (it->size() != 2)
            {
                cmd.getOutput()->usage(cmd);
                ait::log_error() << "Image list file should contain two columns with the data and label filenames.";
                exit(-1);
            }
            const std::string& data_filename = (*it)[0];
            const std::string& label_filename = (*it)[1];

            boost::filesystem::path data_path = boost::filesystem::path(data_filename);
            boost::filesystem::path label_path = boost::filesystem::path(label_filename);
            if (!data_path.is_absolute())
            {
                data_path = boost::filesystem::path(image_list_file).parent_path();
                data_path /= data_filename;
            }
            if (!label_path.is_absolute())
            {
                label_path = boost::filesystem::path(image_list_file).parent_path();
                label_path /= label_filename;
            }

            image_list.push_back(std::make_tuple(data_path.string(), label_path.string()));
        }
        ait::log_info(false) << " Done." << std::endl;

        // TODO: Ensure that label images do not contain values > num_of_classes except for background pixels. Other approach: Test samples directly below.

        // Create weak learner and trainer.
        StatisticsT::Factory statistics_factory(num_of_classes);
        ait::label_type background_label;
        if (background_label_arg.isSet())
        {
            background_label = background_label_arg.getValue();
        }
        else
        {
            background_label = num_of_classes;
        }
        weak_learner_parameters.background_label = background_label;
#if AIT_MULTI_THREADING
        if (num_of_threads_arg.isSet())
        {
            training_parameters.num_of_threads = num_of_threads_arg.getValue();
        }
#endif
        if (save_regularly)
        {
            if (json_forest_file_arg.isSet())
            {
                training_parameters.temporary_json_forest_file_prefix = json_forest_file_arg.getValue();
            }
            else if (binary_forest_file_arg.isSet())
            {
                training_parameters.temporary_binary_forest_file_prefix = binary_forest_file_arg.getValue();
            }
        }
        WeakLearnerT iwl(weak_learner_parameters, statistics_factory);
        ForestTrainerT trainer(iwl, training_parameters);
        SampleProviderT sample_provider(image_list, weak_learner_parameters);
        BaggingWrapperT bagging_wrapper(trainer, sample_provider);

#if AIT_TESTING
        RandomEngineT rnd_engine(11);
#else
        std::random_device rnd_device;
        ait::log_info() << "rnd(): " << rnd_device();
        RandomEngineT rnd_engine(rnd_device());
#endif

        // Train a forest and time it.
        auto start_time = std::chrono::high_resolution_clock::now();
        // TODO
//		ForestTrainerT::ForestT forest = bagging_wrapper.train_forest(rnd_engine);
        // TODO: Testing all samples for comparison with depth_trainer
        sample_provider.clear_samples();
        for (int i = 0; i < image_list.size(); ++i)
        {
        	sample_provider.load_samples_from_image(i, rnd_engine);
        }
        SampleIteratorT samples_start = sample_provider.get_samples_begin();
        SampleIteratorT samples_end = sample_provider.get_samples_end();
        ait::log_info() << "Starting training ...";
        ForestTrainerT::ForestT forest = trainer.train_forest(samples_start, samples_end, rnd_engine);
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = stop_time - start_time;
        auto period = std::chrono::high_resolution_clock::period();
        double elapsed_seconds = duration.count() * period.num / static_cast<double>(period.den);
        ait::log_info() << "Done.";
        ait::log_info() << "Running time: " << elapsed_seconds;

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
        if (print_confusion_matrix)
        {
            ait::log_info(false) << "Creating samples for testing ... " << std::flush;
            sample_provider.clear_samples();
            for (int i = 0; i < image_list.size(); ++i)
            {
            	sample_provider.load_samples_from_image(i, rnd_engine);
            }
            SampleIteratorT samples_start = sample_provider.get_samples_begin();
            SampleIteratorT samples_end = sample_provider.get_samples_end();
            ait::log_info(false) << " Done." << std::endl;
            
            std::vector<ait::size_type> sample_counts(num_of_classes, 0);
            for (auto sample_it = samples_start; sample_it != samples_end; sample_it++)
            {
                ++sample_counts[sample_it->get_label()];
            }
            auto logger = ait::log_info(true);
            logger << "Sample counts>> ";
            for (int c = 0; c < num_of_classes; ++c)
            {
                if (c > 0)
                {
                    logger << ", ";
                }
                logger << "class " << c << ": " << sample_counts[c];
            }
            logger.close();
            // For each tree extract leaf node indices for each sample.
            std::vector<std::vector<ait::size_type>> forest_leaf_indices = forest.evaluate(samples_start, samples_end);

            // Compute number of prediction matches based on a majority vote among the forest.
            int match = 0;
            int no_match = 0;
            for (auto tree_it = forest.cbegin(); tree_it != forest.cend(); ++tree_it)
            {
                for (auto sample_it = samples_start; sample_it != samples_end; sample_it++)
                {
                    const auto &node_it = tree_it->cbegin() + (forest_leaf_indices[tree_it - forest.cbegin()][sample_it - samples_start]);
                    const auto &statistics = node_it->get_statistics();
                    auto max_it = std::max_element(statistics.get_histogram().cbegin(), statistics.get_histogram().cend());
                    auto label = max_it - statistics.get_histogram().cbegin();
                    if (label == sample_it->get_label())
                        match++;
                    else
                        no_match++;
                }
            }
            ait::log_info() << "Match: " << match << ", no match: " << no_match;
            // Compute confusion matrix.
            auto forest_utils = ait::make_forest_utils(forest);
            auto confusion_matrix = forest_utils.compute_confusion_matrix(samples_start, samples_end);
            ait::log_info() << "Confusion matrix:" << std::endl << confusion_matrix;
            auto norm_confusion_matrix = ait::EvaluationUtils::normalize_confusion_matrix(confusion_matrix);
            ait::log_info() << "Normalized confusion matrix:" << std::endl << norm_confusion_matrix;
            ait::log_info() << "Diagonal of normalized confusion matrix:" << std::endl << norm_confusion_matrix.diagonal();
            
            // Computing per-frame confusion matrix
            ait::log_info() << "Computing per-frame confusion matrix.";
            using ConfusionMatrixType = typename decltype(forest_utils)::MatrixType;
            ConfusionMatrixType per_frame_confusion_matrix(num_of_classes, num_of_classes);
            per_frame_confusion_matrix.setZero();
            WeakLearnerT::ParametersT full_parameters(weak_learner_parameters);
            // Modify parameters to retrieve all pixels per sample
            full_parameters.samples_per_image_fraction = 1.0;
            SampleProviderT full_sample_provider(image_list, full_parameters);
            for (int i = 0; i < image_list.size(); ++i) {
                full_sample_provider.clear_samples();
                full_sample_provider.load_samples_from_image(i, rnd_engine);
                samples_start = full_sample_provider.get_samples_begin();
                samples_end = full_sample_provider.get_samples_end();
                forest_utils.update_confusion_matrix(per_frame_confusion_matrix, samples_start, samples_end);
            }
            ait::log_info() << "Per-frame confusion matrix:" << std::endl << per_frame_confusion_matrix;
            ConfusionMatrixType per_frame_norm_confusion_matrix = ait::EvaluationUtils::normalize_confusion_matrix(per_frame_confusion_matrix);
            ait::log_info() << "Normalized per-frame confusion matrix:" << std::endl << per_frame_norm_confusion_matrix;
            ait::log_info() << "Diagonal of normalized per-frame confusion matrix:" << std::endl << per_frame_norm_confusion_matrix.diagonal();
            ait::log_info() << "Mean of diagonal of normalized per-frame confusion matrix:" << std::endl << per_frame_norm_confusion_matrix.diagonal().mean();
        }
    }
    catch (const TCLAP::ArgException &e)
    {
        ait::log_error() << "Error parsing command line: " << e.error() << " for arg " << e.argId();
    }

    return 0;
}

