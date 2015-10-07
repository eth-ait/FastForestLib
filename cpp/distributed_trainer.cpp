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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "distributed_forest_trainer.h"
#include "image_weak_learner.h"
#include "matlab_file_io.h"

using SampleT = ait::ImageSample;
using StatisticsT = ait::HistogramStatistics<SampleT>;
using RandomEngineT = std::mt19937_64;

using SampleContainerT = std::vector<const SampleT>;
using SampleIteratorT= typename SampleContainerT::const_iterator;

template <class TSampleIterator, class TRandomEngine> using WeakLearnerAliasT
= typename ait::ImageWeakLearner<StatisticsT::Factory, TSampleIterator, TRandomEngine>;

using ForestTrainerT = ait::DistributedForestTrainer<WeakLearnerAliasT, SampleIteratorT, RandomEngineT>;
using WeakLearnerT = typename ForestTrainerT::WeakLearnerT;

namespace mpi = boost::mpi;

std::string mpi_output_prefix(const mpi::communicator &comm)
{
    std::ostringstream out;
    out << "[" << comm.rank() << "] ";
    return out.str();
}

int main(int argc, const char *argv[]) {
    try {
        // Initialize MPI.
        mpi::environment env;
        mpi::communicator world;
        std::cout << "Rank " << world.rank() << " of " << world.size() << "." << std::endl;

        // Parse command line arguments.
        TCLAP::CmdLine cmd("Distributed RF trainer", ' ', "0.3");
        TCLAP::ValueArg<std::string> data_file_arg("d", "data-file", "File containing image data", true, "", "string", cmd);
        TCLAP::ValueArg<std::string> text_forest_file_arg("t", "text-forest-file", "Text file where the trained forest should be saved", false, "forest.txt", "string", cmd);
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string", cmd);
        TCLAP::SwitchArg print_confusion_matrix_switch("c", "conf-matrix", "Print confusion matrix", cmd, true);
        cmd.parse(argc, argv);
        
        std::string data_file = data_file_arg.getValue();
        bool print_confusion_matrix = print_confusion_matrix_switch.getValue();
        
        // Read data from file.
        std::cout << "Reading images ... " << std::flush;
        std::vector<ait::Image> images = ait::load_images_from_matlab_file(data_file, "data", "labels");
        std::cout << " Done." << std::endl;
        
        // Compute number of classes from data.
        std::cout << "Computing number of classes ..." << std::flush;
        ait::label_type max_label = 0;
        for (auto i = 0; i < images.size(); i++) {
            ait::label_type local_max_label = images[i].get_label_matrix().maxCoeff();
            max_label = std::max(local_max_label, max_label);
        }
        ait::size_type num_of_classes = static_cast<ait::size_type>(max_label) + 1;
        std::cout << " Found " << num_of_classes << " classes." << std::endl;
        
        // Extract samples from data.
        std::cout << "Creating samples ... " << std::flush;
        SampleContainerT samples;
        for (auto i = 0; i < images.size(); i++) {
            if ((i - world.rank()) % world.size() == 0)
            {
                for (int x=0; x < images[i].get_data_matrix().rows(); x++)
                {
                    if (x % 8 != 0)
                        continue;
//                    for (int y=0; y < images[i].get_data_matrix().cols(); y++)
                    {
                        int y = 0;
                        SampleT sample = ait::ImageSample(&images[i], x, y);
//                        ImageSamplePointerT sample_ptr = std::make_shared<SampleT>(&images[i], x, y);
                        samples.push_back(sample);
                    }
                }
            }
        }
        std::cout << " Done." << std::endl;

        // Create weak learner and trainer.
        StatisticsT::Factory statistics_factory(num_of_classes);
        WeakLearnerT::ParametersT weak_learner_parameters;
        ForestTrainerT::ParametersT training_parameters;
        WeakLearnerT iwl(weak_learner_parameters, statistics_factory);
        ForestTrainerT trainer(world, iwl, training_parameters);

        // Train a forest and time it.
        auto start_time = std::chrono::high_resolution_clock::now();
        ForestTrainerT::ForestT forest = trainer.train_forest(samples.cbegin(), samples.cend());
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = stop_time - start_time;
        auto period = std::chrono::high_resolution_clock::period();
        double elapsed_seconds = duration.count() * period.num / static_cast<double>(period.den);
        std::cout << "Running time: " << elapsed_seconds << std::endl;
        
        // Optionally: Serialize forest to text file.
        if (text_forest_file_arg.isSet())
        {
            {
                std::cout << "Writing json forest file " << text_forest_file_arg.getValue() << "... " << std::flush;
                std::ofstream ofile(text_forest_file_arg.getValue());
                boost::archive::text_oarchive oarchive(ofile);
                oarchive << forest;
                std::cout << " Done." << std::endl;
            }
            
            {
                // Read forest from file for testing
                std::cout << "Reading json forest file " << text_forest_file_arg.getValue() << " ... " << std::flush;
                std::ifstream ifile(text_forest_file_arg.getValue());
                boost::archive::text_iarchive iarchive(ifile);
                iarchive >> forest;
                std::cout << " Done." << std::endl;
            }
        }
        
        // Optionally: Serialize forest to binary file.
        if (binary_forest_file_arg.isSet())
        {
            {
                // Serialize forest to file
                std::cout << "Writing binary forest file " << binary_forest_file_arg.getValue() << "... " << std::flush;
                std::ofstream ofile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                boost::archive::binary_oarchive oarchive(ofile);
                oarchive << forest;
                std::cout << " Done." << std::endl;
            }
            
            {
                // Read forest from file for testing
                std::cout << "Reading binary forest file " << binary_forest_file_arg.getValue() << " ... " << std::flush;
                std::ifstream ifile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                boost::archive::binary_iarchive iarchive(ifile);
                iarchive >> forest;
                std::cout << " Done." << std::endl;
            }
        }
        
        // Optionally: Compute some stats and print them.
        if (print_confusion_matrix)
        {
            // For each tree extract leaf node indices for each sample.
            std::vector<std::vector<ait::size_type>> forest_leaf_indices = forest.evaluate(samples.cbegin(), samples.cend());
            
            // Compute number of prediction matches based on a majority vote among the forest.
            int match = 0;
            int no_match = 0;
            for (auto tree_it = forest.cbegin(); tree_it != forest.cend(); ++tree_it)
            {
                for (auto sample_it=samples.cbegin(); sample_it != samples.cend(); sample_it++)
                {
                    const auto &node_it = tree_it->cbegin() + (forest_leaf_indices[tree_it - forest.cbegin()][sample_it - samples.cbegin()]);
                    const auto &statistics = node_it->get_statistics();
                    auto max_it = std::max_element(statistics.get_histogram().cbegin(), statistics.get_histogram().cend());
                    auto label = max_it - statistics.get_histogram().cbegin();
                    if (label == sample_it->get_label())
                        match++;
                    else
                        no_match++;
                }
            }
            std::cout << "Match: " << match << ", no match: " << no_match << std::endl;
            
            // Compute confusion matrix.
            auto tree_utils = ait::make_tree_utils<SampleIteratorT>(*forest.begin());
            auto matrix = tree_utils.compute_confusion_matrix(num_of_classes, samples.cbegin(), samples.cend());
            std::cout << "Confusion matrix:" << std::endl << matrix << std::endl;
            auto norm_matrix = tree_utils.compute_normalized_confusion_matrix(num_of_classes, samples.cbegin(), samples.cend());
            std::cout << "Normalized confusion matrix:" << std::endl << norm_matrix << std::endl;
        }
    }
    catch (const TCLAP::ArgException &e)
    {
        std::cerr << "Error parsing command line: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}
