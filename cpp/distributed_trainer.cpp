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
#include <boost/filesystem/path.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "distributed_forest_trainer.h"
#include "distributed_bagging_wrapper.h"
#include "image_weak_learner.h"
#include "csv_utils.h"

// TODO: Distributed bagging sample provider

using PixelT = ait::pixel_type;
using ImageT = ait::Image<PixelT>;
using SampleT = ait::ImageSample<PixelT>;
using StatisticsT = ait::HistogramStatistics<SampleT>;
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
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", true, "", "string", cmd);
        TCLAP::ValueArg<int> num_of_classes_arg("n", "num-of-classes", "Number of classes in the data", true, 1, "int", cmd);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string", cmd);
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string", cmd);
        TCLAP::SwitchArg hide_confusion_matrix_switch("c", "no-conf-matrix", "Don't print confusion matrix", cmd, false);
        cmd.parse(argc, argv);

        const int num_of_classes = num_of_classes_arg.getValue();
        const bool print_confusion_matrix = !hide_confusion_matrix_switch.getValue();
        const std::string image_list_file = image_list_file_arg.getValue();

        // Read image file list
        ait::log_info(false) << "Reading image list ... " << std::flush;
        std::vector<std::tuple<std::string, std::string>> image_list;
        std::ifstream ifile(image_list_file);
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

        // Create weak learner and trainer.
        StatisticsT::Factory statistics_factory(num_of_classes);
        WeakLearnerT::ParametersT weak_learner_parameters;
        ForestTrainerT::ParametersT training_parameters;
        WeakLearnerT iwl(weak_learner_parameters, statistics_factory);
        ForestTrainerT trainer(world, iwl, training_parameters);
        SampleProviderT sample_provider(image_list, weak_learner_parameters);
        BaggingWrapperT bagging_wrapper(world, trainer, sample_provider);
#ifdef AIT_TESTING
        RandomEngineT rnd_engine(11);
#else
        std::random_device rnd_device;
        ait::log_info() << "rnd(): " << rnd_device();
        RandomEngineT rnd_engine(rnd_device());
#endif

        // Train a forest and time it.
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
            if (binary_forest_file_arg.isSet())
            {
                {
                    ait::log_info(false) << "Writing binary forest file " << binary_forest_file_arg.getValue() << "... " << std::flush;
                    std::ofstream ofile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                    cereal::BinaryOutputArchive oarchive(ofile);
                    oarchive(cereal::make_nvp("forest", forest));
                    ait::log_info(false) << " Done." << std::endl;
                }
            }

            // Optionally: Compute some stats and print them.
            if (print_confusion_matrix && world.rank() == 0)
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
                        const auto& node_it = tree_it->cbegin() + (forest_leaf_indices[tree_it - forest.cbegin()][sample_it - samples_start]);
                        const auto& statistics = node_it->get_statistics();
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
                auto tree_utils = ait::make_tree_utils<SampleIteratorT>(*forest.begin());
                auto matrix = tree_utils.compute_confusion_matrix(num_of_classes, samples_start, samples_end);
                ait::log_info() << "Confusion matrix:" << std::endl << matrix;
                auto norm_matrix = tree_utils.compute_normalized_confusion_matrix(num_of_classes, samples_start, samples_end);
                ait::log_info() << "Normalized confusion matrix:" << std::endl << norm_matrix;
            }
        }
    }
    catch (const TCLAP::ArgException& e)
    {
        ait::log_error() << "Error parsing command line: " << e.error() << " for arg " << e.argId();
    }

    return 0;
}
