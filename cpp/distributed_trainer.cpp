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

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/filesystem/path.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "distributed_forest_trainer.h"
#include "image_weak_learner.h"
#include "csv_utils.h"
#include "matlab_file_io.h"

// TODO: Distributed bagging sample provider

using PixelT = ait::pixel_type;
using ImageT = ait::Image<PixelT>;
using SampleT = ait::ImageSample<PixelT>;
using StatisticsT = ait::HistogramStatistics<SampleT>;
using RandomEngineT = std::mt19937_64;

using SampleContainerT = std::vector<SampleT>;
using SampleIteratorT= typename SampleContainerT::const_iterator;

template <typename TSampleIterator, typename TRandomEngine> using WeakLearnerAliasT
    = typename ait::ImageWeakLearner<StatisticsT::Factory, TSampleIterator, TRandomEngine, PixelT>;

using ForestTrainerT = ait::DistributedForestTrainer<WeakLearnerAliasT, SampleIteratorT, RandomEngineT>;
using WeakLearnerT = typename ForestTrainerT::WeakLearnerT;
//using BaggingWrapperT = BaggingWrapper<ForestTrainerT>;

namespace mpi = boost::mpi;

int main(int argc, const char *argv[]) {
    try {
        // Initialize MPI.
        mpi::environment env;
        mpi::communicator world;
        std::ostringstream prefix_stream;
        prefix_stream << world.rank() << "> ";
        ait::logger().set_prefix(prefix_stream.str());
        ait::log_info() << "Rank " << world.rank() << " of " << world.size() << ".";

        // Parse command line arguments.
        TCLAP::CmdLine cmd("Distributed RF trainer", ' ', "0.3");
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", true, "", "string", cmd);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string", cmd);
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string", cmd);
        TCLAP::SwitchArg print_confusion_matrix_switch("c", "conf-matrix", "Print confusion matrix", cmd, true);
        cmd.parse(argc, argv);

        // Read image file list
        ait::log_info(false) << "Reading image list ... " << std::flush;
        std::vector<std::tuple<std::string, std::string>> image_list;
        std::ifstream ifile(image_list_file_arg.getValue());
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
            image_list.push_back(std::make_tuple(data_filename, label_filename));
        }
        ait::log_info(false) << " Done." << std::endl;

        // Read data from file.
        ait::log_info(false) << "Reading images ... " << std::flush;
        std::vector<ImageT> images;
        for (auto it = image_list.cbegin(); it != image_list.cend(); ++it)
        {
            boost::filesystem::path data_path;
            boost::filesystem::path label_path = boost::filesystem::path(std::get<1>(*it));
            if (data_path.is_absolute())
            {
                data_path = boost::filesystem::path(std::get<0>(*it));
            }
            else
            {
                data_path = boost::filesystem::path(image_list_file_arg.getValue()).parent_path();
                data_path /= std::get<0>(*it);
            }
            if (label_path.is_absolute())
            {
                label_path = boost::filesystem::path(std::get<1>(*it));
            }
            else
            {
                label_path = boost::filesystem::path(image_list_file_arg.getValue()).parent_path();
                label_path /= std::get<1>(*it);
            }
            ImageT image = ImageT::load_from_files(data_path.string(), label_path.string());
            images.push_back(std::move(image));
        }
        ait::log_info(false) << " Done." << std::endl;

        // Compute number of classes from data.
        ait::log_info(false) << "Computing number of classes ..." << std::flush;
        ait::label_type max_label = 0;
        for (auto i = 0; i < images.size(); i++) {
            ait::label_type local_max_label = images[i].get_label_matrix().maxCoeff();
            max_label = std::max(local_max_label, max_label);
        }
        ait::size_type num_of_classes = static_cast<ait::size_type>(max_label) + 1;
        ait::log_info(false) << " Found " << num_of_classes << " classes." << std::endl;

        // Extract samples from data.
        ait::log_info(false) << "Creating samples ... " << std::flush;
        SampleContainerT samples;
        for (auto i = 0; i < images.size(); i++) {
            if ((i - world.rank()) % world.size() == 0)
            {
                for (int x=0; x < images[i].get_data_matrix().rows(); x++)
                {
//                    if (x % 8 != 0)
//                        continue;
                    for (int y=0; y < images[i].get_data_matrix().cols(); y++)
                    {
//                        int y = 0;
                        SampleT sample = SampleT(&images[i], x, y);
//                        ImageSamplePointerT sample_ptr = std::make_shared<SampleT>(&images[i], x, y);
                        samples.push_back(sample);
                    }
                }
            }
        }
        ait::log_info(false) << " Done." << std::endl;

        // Create weak learner and trainer.
        StatisticsT::Factory statistics_factory(num_of_classes);
        WeakLearnerT::ParametersT weak_learner_parameters;
        ForestTrainerT::ParametersT training_parameters;
        WeakLearnerT iwl(weak_learner_parameters, statistics_factory);
        ForestTrainerT trainer(world, iwl, training_parameters);
#ifdef AIT_TESTING
        RandomEngineT rnd_engine(11);
#else
        std::random_device rnd_device;
        ait::log_info() << "rnd(): " << rnd_device();
        RandomEngineT rnd_engine(rnd_device());
#endif

        // Train a forest and time it.
        auto start_time = std::chrono::high_resolution_clock::now();
        ForestTrainerT::ForestT forest = trainer.train_forest(samples.cbegin(), samples.cend(), rnd_engine);
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
            if (print_confusion_matrix_switch.getValue())
            {
                // Extract samples from data.
                ait::log_info(false) << "Creating samples ... " << std::flush;
                samples.clear();
                for (auto i = 0; i < images.size(); i++) {
                    for (int x=0; x < images[i].get_data_matrix().rows(); x++)
                    {
                        for (int y=0; y < images[i].get_data_matrix().cols(); y++)
                        {
                            SampleT sample = SampleT(&images[i], x, y);
                            samples.push_back(sample);
                        }
                    }
                }
                ait::log_info(false) << " Done." << std::endl;

                // For each tree extract leaf node indices for each sample.
                std::vector<std::vector<ait::size_type>> forest_leaf_indices = forest.evaluate(samples.cbegin(), samples.cend());
                
                // Compute number of prediction matches based on a majority vote among the forest.
                int match = 0;
                int no_match = 0;
                for (auto tree_it = forest.cbegin(); tree_it != forest.cend(); ++tree_it)
                {
                    for (auto sample_it=samples.cbegin(); sample_it != samples.cend(); sample_it++)
                    {
                        const auto& node_it = tree_it->cbegin() + (forest_leaf_indices[tree_it - forest.cbegin()][sample_it - samples.cbegin()]);
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
                auto matrix = tree_utils.compute_confusion_matrix(num_of_classes, samples.cbegin(), samples.cend());
                ait::log_info() << "Confusion matrix:" << std::endl << matrix;
                auto norm_matrix = tree_utils.compute_normalized_confusion_matrix(num_of_classes, samples.cbegin(), samples.cend());
                ait::log_info() << "Normalized confusion matrix:" << std::endl << norm_matrix;
            }
        }
    }
    catch (const TCLAP::ArgException& e)
    {
        std::cerr << "Error parsing command line: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}
