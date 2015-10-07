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

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "level_forest_trainer.h"
#include "image_weak_learner.h"
#include "matlab_file_io.h"

int main(int argc, const char *argv[]) {
    try {
        TCLAP::CmdLine cmd("Level RF trainer", ' ', "0.3");
        TCLAP::ValueArg<std::string> data_file_arg("d", "data-file", "File containing image data", true, "", "string", cmd);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string", cmd);
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string", cmd);
        TCLAP::SwitchArg print_confusion_matrix_switch("c", "conf-matrix", "Print confusion matrix", cmd, true);
        cmd.parse(argc, argv);

        std::string data_file = data_file_arg.getValue();
        bool print_confusion_matrix = print_confusion_matrix_switch.getValue();

        std::cout << "Reading images ... " << std::flush;
        std::vector<ait::Image> images = ait::load_images_from_matlab_file(data_file, "data", "labels");
        std::cout << " Done." << std::endl;

        std::cout << "Computing number of classes ..." << std::flush;
        ait::size_type num_of_classes = 0;
        for (auto i = 0; i < images.size(); i++) {
            ait::label_type max_label = images[i].get_label_matrix().maxCoeff();
            num_of_classes = std::max(static_cast<ait::size_type>(max_label) + 1, num_of_classes);
        }
        std::cout << " Found " << num_of_classes << " classes." << std::endl;

        std::cout << "Creating samples ... " << std::flush;
        using ImageSamplePointer = std::shared_ptr<ait::ImageSample>;
        std::vector<ImageSamplePointer> samples;
        for (auto i = 0; i < images.size(); i++) {
            for (int x=0; x < images[i].get_data_matrix().rows(); x++)
            {
                if (x % 8 != 0)
                    continue;
//                for (int y=0; y < images[i].get_data_matrix().cols(); y++)
                {
                    int y = 0;
                    ImageSamplePointer sample_ptr = std::make_shared<ait::ImageSample>(&images[i], x, y);
                    samples.push_back(sample_ptr);
                }
            }
        }
        std::cout << " Done." << std::endl;

        using SamplePointerIteratorType = std::vector<ImageSamplePointer>::const_iterator;
        using SampleIteratorType = ait::PointerIteratorWrapper<SamplePointerIteratorType>;
        using StatisticsFactoryType = typename ait::HistogramStatistics<ait::ImageSample>::HistogramStatisticsFactory;
        using WeakLearnerType = ait::ImageWeakLearner<StatisticsFactoryType, SampleIteratorType>;
        using RandomEngine = std::mt19937_64;
        
        StatisticsFactoryType statistics_factory(num_of_classes);
        ait::ImageWeakLearnerParameters weak_learner_parameters;
        ait::LevelTrainingParameters training_parameters;
        WeakLearnerType iwl(weak_learner_parameters, statistics_factory);
        
        ait::LevelForestTrainer<SamplePointerIteratorType, WeakLearnerType, RandomEngine> trainer(iwl, training_parameters);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        using ForestType = ait::Forest<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample> >;
        ForestType forest = trainer.train_forest(samples.cbegin(), samples.cend());
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = stop_time - start_time;
        auto period = std::chrono::high_resolution_clock::period();
        double elapsed_seconds = duration.count() * period.num / static_cast<double>(period.den);
        std::cout << "Running time: " << elapsed_seconds << std::endl;

        if (json_forest_file_arg.isSet())
        {
            {
                // Serialize forest to file
                std::cout << "Writing json forest file " << json_forest_file_arg.getValue() << "... " << std::flush;
                std::ofstream ofile(json_forest_file_arg.getValue());
                cereal::JSONOutputArchive oarchive(ofile);
                oarchive(cereal::make_nvp("forest", forest));
                std::cout << " Done." << std::endl;
            }

            {
                // Read forest from file for testing
                std::cout << "Reading json forest file " << json_forest_file_arg.getValue() << " ... " << std::flush;
                std::ifstream ifile(json_forest_file_arg.getValue());
                cereal::JSONInputArchive iarchive(ifile);
                iarchive(forest);
                std::cout << " Done." << std::endl;
            }
        }
        
        if (binary_forest_file_arg.isSet())
        {
            {
                // Serialize forest to file
                std::cout << "Writing binary forest file " << binary_forest_file_arg.getValue() << "... " << std::flush;
                std::ofstream ofile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                cereal::BinaryOutputArchive oarchive(ofile);
                oarchive(cereal::make_nvp("forest", forest));
                std::cout << " Done." << std::endl;
            }
            
            {
                // Read forest from file for testing
                std::cout << "Reading binary forest file " << binary_forest_file_arg.getValue() << " ... " << std::flush;
                std::ifstream ifile(binary_forest_file_arg.getValue(), std::ios_base::binary);
                cereal::BinaryInputArchive iarchive(ifile);
                iarchive(forest);
                std::cout << " Done." << std::endl;
            }
        }

        if (print_confusion_matrix)
        {
            auto samples_start = ait::make_pointer_iterator_wrapper(samples.cbegin());
            auto samples_end = ait::make_pointer_iterator_wrapper(samples.cend());
            std::vector<std::vector<ait::size_type> > forest_leaf_indices = forest.evaluate<SampleIteratorType>(samples_start, samples_end);

            int match = 0;
            int no_match = 0;
            for (std::size_t i=0; i < forest.size(); i++)
            {
                const ForestType::TreeType &tree = forest.get_tree(i);
                for (auto it=samples.cbegin(); it != samples.cend(); it++)
                {
                    const auto &node_it = tree.cbegin() + (forest_leaf_indices[i][it - samples.cbegin()]);
                    const auto &statistics = node_it->get_statistics();
                    auto max_it = std::max_element(statistics.get_histogram().cbegin(), statistics.get_histogram().cend());
                    auto label = max_it - statistics.get_histogram().cbegin();
                    if (label == (*it)->get_label())
                        match++;
                    else
                        no_match++;
                }
            }
            std::cout << "Match: " << match << ", no match: " << no_match << std::endl;

            ait::Tree<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample> > tree = forest.get_tree(0);
            ait::TreeUtilities<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample>, SampleIteratorType> tree_utils(tree);
            auto matrix = tree_utils.compute_confusion_matrix<3>(samples_start, samples_end);
            std::cout << "Confusion matrix:" << std::endl << matrix << std::endl;
            auto norm_matrix = tree_utils.compute_normalized_confusion_matrix<3>(samples_start, samples_end);
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

