//
//  depth_forest_trainer_mat.cpp
//  DistRandomForest
//
//  Created by Benjamin Hepp on 04/02/16.
//
//

#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <iostream>
#include <chrono>
#include <limits>

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "depth_forest_trainer.h"
#include "histogram_statistics.h"
#include "image_weak_learner.h"
#include "eigen_matrix_io.h"
#include "matlab_file_io.h"
#include "evaluation_utils.h"

using ImageT = ait::Image<>;
using SampleT = ait::ImageSample<>;
using StatisticsT = ait::HistogramStatistics;
using RandomEngineT = std::mt19937_64;

using SampleContainerT = std::vector<SampleT>;
using SampleIteratorT= typename SampleContainerT::iterator;
using ConstSampleIteratorT= typename SampleContainerT::const_iterator;

template <class TSampleIterator> using WeakLearnerAliasT
= typename ait::ImageWeakLearner<StatisticsT::Factory, TSampleIterator, RandomEngineT>;

using ForestTrainerT = ait::DepthForestTrainer<WeakLearnerAliasT, SampleIteratorT>;
using WeakLearnerT = typename ForestTrainerT::WeakLearnerT;

int main(int argc, const char* argv[]) {
    try {
        // Parse command line arguments.
        TCLAP::CmdLine cmd("Depth RF trainer", ' ', "0.3");
        TCLAP::ValueArg<std::string> data_file_arg("d", "data-file", "File containing image data", true, "", "string", cmd);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string", cmd);
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string", cmd);
        TCLAP::ValueArg<int> background_label_arg("l", "background-label", "Label of background pixels to be ignored", false, -1, "int", cmd);
        TCLAP::SwitchArg print_confusion_matrix_switch("c", "conf-matrix", "Print confusion matrix", cmd, true);
        cmd.parse(argc, argv);
        
        std::string data_file = data_file_arg.getValue();
        bool print_confusion_matrix = print_confusion_matrix_switch.getValue();
        
        // Read data from file.
        ait::log_info(false) << "Reading images ... " << std::flush;
        std::vector<ImageT> images = ait::load_images_from_matlab_file(data_file, "data", "labels");
        ait::log_info(false) << " Done." << std::endl;
        
        // Print size of images
        ait::size_type image_height = images[0].get_data_matrix().rows();
        ait::size_type image_width = images[0].get_data_matrix().cols();
        ait::log_info(false) << "Image size " << image_width << " x " << image_height << std::endl;
        
        // Compute number of classes from data.
        ait::log_info(false) << "Computing number of classes ..." << std::flush;
        ait::label_type max_label = 0;
        for (auto i = 0; i < images.size(); i++) {
            ait::label_type local_max_label = images[i].get_label_matrix().maxCoeff();
            max_label = std::max(local_max_label, max_label);
        }
        ait::size_type num_of_classes = static_cast<ait::size_type>(max_label) + 1;
        ait::log_info(false) << " Found " << num_of_classes << " classes." << std::endl;
        
        // Compute value range of data
        ait::log_info(false) << "Computing value range of data ..." << std::flush;
        ait::pixel_type max_value = std::numeric_limits<ait::pixel_type>::min();
        ait::pixel_type min_value = std::numeric_limits<ait::pixel_type>::max();
        for (auto i = 0; i < images.size(); i++) {
            ait::label_type local_min_value = images[i].get_data_matrix().minCoeff();
            min_value = std::min(local_min_value, min_value);
            ait::label_type local_max_value = images[i].get_data_matrix().maxCoeff();
            max_value = std::max(local_max_value, max_value);
        }
        ait::log_info(false) << " Value range [" << min_value << ", " << max_value << "]." << std::endl;

        // Extract samples from data.
        ait::log_info(false) << "Creating samples ... " << std::flush;
        SampleContainerT samples;
        for (auto i = 0; i < images.size(); i++) {
            for (int x=0; x < images[i].get_data_matrix().rows(); x++)
            {
                //                if (x % 8 != 0)
                //                    continue;
                for (int y=0; y < images[i].get_data_matrix().cols(); y++)
                {
                    //                    int y = 0;
                    if (background_label_arg.isSet())
                    {
                        if (images[i].get_label_matrix()(x, y) == background_label_arg.getValue())
                        {
                            continue;
                        }
                    }
                    SampleT sample = SampleT(&images[i], x, y);
                    samples.push_back(sample);
                }
            }
        }
        ait::log_info(false) << " Done." << std::endl;
        
        // Create weak learner and trainer.
        StatisticsT::Factory statistics_factory(num_of_classes);
        WeakLearnerT::ParametersT weak_learner_parameters;
        ForestTrainerT::ParametersT training_parameters;
        WeakLearnerT iwl(weak_learner_parameters, statistics_factory);
        ForestTrainerT trainer(iwl, training_parameters);
#ifdef AIT_TESTING
        RandomEngineT rnd_engine(13);
#else
        std::random_device rnd_device;
        ait::log_info() << "rnd(): " << rnd_device();
        RandomEngineT rnd_engine(rnd_device());
#endif
        
        // Train a forest and time it.
        // TODO
        auto start_time = std::chrono::high_resolution_clock::now();
        ForestTrainerT::ForestT forest = trainer.train_forest(samples.begin(), samples.end(), rnd_engine);
//        ForestTrainerT::ForestT forest = trainer.train_forest(samples.begin(), samples.end(), rnd_engine);
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = stop_time - start_time;
        auto period = std::chrono::high_resolution_clock::period();
        double elapsed_seconds = duration.count() * period.num / static_cast<double>(period.den);
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
        if (print_confusion_matrix)
        {
            std::vector<ait::size_type> sample_counts(num_of_classes, 0);
            for (auto sample_it = samples.cbegin(); sample_it != samples.cend(); sample_it++)
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
            auto forest_utils = ait::make_forest_utils(forest);
            auto matrix = forest_utils.compute_confusion_matrix(samples.cbegin(), samples.cend());
            ait::log_info() << "Confusion matrix:" << std::endl << matrix;
            auto norm_matrix = ait::EvaluationUtils::normalize_confusion_matrix(matrix);
            ait::log_info() << "Normalized confusion matrix:" << std::endl << norm_matrix;
            ait::log_info() << "Diagonal of normalized confusion matrix:" << std::endl << norm_matrix.diagonal();
        }
    }
    catch (const std::runtime_error& error)
    {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}
