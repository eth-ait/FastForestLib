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

using PixelT = ait::pixel_type;
using ImageT = ait::Image<PixelT>;
using SampleT = ait::ImageSample<PixelT>;
using StatisticsT = ait::HistogramStatistics<SampleT>;
using SplitPointT = ait::ImageSplitPoint<PixelT>;
using RandomEngineT = std::mt19937_64;

using ForestT = ait::Forest<SplitPointT, StatisticsT>;
using SampleProviderT = ait::ImageSampleProvider<RandomEngineT>;
using ParametersT = typename SampleProviderT::ParametersT;
using SampleIteratorT = typename SampleProviderT::SampleIteratorT;

int main(int argc, const char* argv[]) {
    try {
        // Parse command line arguments.
        TCLAP::CmdLine cmd("Random forest predictor", ' ', "0.3");
        TCLAP::ValueArg<std::string> image_list_file_arg("f", "image-list-file", "File containing the names of image files", true, "", "string", cmd);
        TCLAP::ValueArg<int> num_of_classes_arg("n", "num-of-classes", "Number of classes in the data", true, 1, "int", cmd);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file of the forest to load", false, "forest.json", "string");
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file of the forest to load", false, "forest.bin", "string");
        TCLAP::ValueArg<int> background_label_arg("l", "background-label", "Lower bound of background labels to be ignored", false, -1, "int", cmd);
        cmd.xorAdd(json_forest_file_arg, binary_forest_file_arg);
        cmd.parse(argc, argv);
        
        const int num_of_classes = num_of_classes_arg.getValue();
        const std::string image_list_file = image_list_file_arg.getValue();
        
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

        ForestT forest;
        // Read forest from JSON file.
        if (json_forest_file_arg.isSet())
        {
            {
                ait::log_info(false) << "Reading json forest file " << json_forest_file_arg.getValue() << "... " << std::flush;
                std::ifstream ifile(json_forest_file_arg.getValue());
                cereal::JSONInputArchive iarchive(ifile);
                iarchive(cereal::make_nvp("forest", forest));
                ait::log_info(false) << " Done." << std::endl;
            }
        }
        // Read forest from binary file.
        else if (binary_forest_file_arg.isSet())
        {
            {
                ait::log_info(false) << "Reading binary forest file " << binary_forest_file_arg.getValue() << "... " << std::flush;
                std::ifstream ifile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                cereal::BinaryInputArchive iarchive(ifile);
                iarchive(cereal::make_nvp("forest", forest));
                ait::log_info(false) << " Done." << std::endl;
            }
        }
        else
        {
            throw("This should never happen. Either a JSON or a binary forest file have to be specified!");
        }


#if AIT_TESTING
        RandomEngineT rnd_engine(11);
#else
        std::random_device rnd_device;
        ait::log_info() << "rnd(): " << rnd_device();
        RandomEngineT rnd_engine(rnd_device());
#endif

        // Compute some stats and print them.
        ait::log_info(false) << "Creating samples for testing ... " << std::flush;
        ParametersT parameters;
        ait::label_type background_label;
        if (background_label_arg.isSet())
        {
            background_label = background_label_arg.getValue();
        }
        else
        {
            background_label = num_of_classes;
        }
        parameters.background_label = background_label;
        SampleProviderT sample_provider(image_list, parameters);
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

        auto tree_utils = ait::make_forest_utils<SampleIteratorT>(*forest.begin());
        auto matrix = tree_utils.compute_confusion_matrix(num_of_classes, samples_start, samples_end);
        ait::log_info() << "Confusion matrix:" << std::endl << matrix;
        auto norm_matrix = tree_utils.normalize_confusion_matrix(confusion_matrix);
        ait::log_info() << "Normalized confusion matrix:" << std::endl << norm_matrix;
        ait::log_info() << "Diagonal of normalized confusion matrix:" << std::endl << norm_matrix.diagonal();

//        auto forest_utils = ait::make_forest_utils<SampleIteratorT>(forest);
//        auto matrix = forest_utils.compute_confusion_matrix(num_of_classes, samples_start, samples_end);
//        ait::log_info() << "Confusion matrix:" << std::endl << matrix;
//        auto norm_matrix = forest_utils.normalize_confusion_matrix(confusion_matrix);
//        ait::log_info() << "Normalized confusion matrix:" << std::endl << norm_matrix;
//        ait::log_info() << "Diagonal of normalized confusion matrix:" << std::endl << norm_matrix.diagonal();
    }
    catch (const TCLAP::ArgException &e)
    {
        ait::log_error() << "Error parsing command line: " << e.error() << " for arg " << e.argId();
    }
    
    return 0;
}

