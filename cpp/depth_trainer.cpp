#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <iostream>
#include <chrono>

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <tclap/CmdLine.h>

#include "ait.h"
#include "depth_forest_trainer.h"
#include "histogram_statistics.h"
#include "image_weak_learner.h"
#include "eigen_matrix_io.h"
#include "matlab_file_io.h"


int main(int argc, const char *argv[]) {
    try {
        TCLAP::CmdLine cmd("Depth RF trainer", ' ', "0.3");
        TCLAP::ValueArg<std::string> data_file_arg("d", "data-file", "File containing image data", true, "", "string", cmd);
        TCLAP::ValueArg<std::string> json_forest_file_arg("j", "json-forest-file", "JSON file where the trained forest should be saved", false, "forest.json", "string", cmd);
        TCLAP::ValueArg<std::string> binary_forest_file_arg("b", "binary-forest-file", "Binary file where the trained forest should be saved", false, "forest.bin", "string", cmd);
        TCLAP::SwitchArg print_confusion_matrix_switch("c", "conf-matrix", "Print confusion matrix", cmd, true);
        cmd.parse(argc, argv);
        
        std::string data_file = data_file_arg.getValue();
        bool print_confusion_matrix = print_confusion_matrix_switch.getValue();

        //		std::vector<std::string> array_names;
        //		array_names.push_back("data");
        //		array_names.push_back("labels");
        //		std::map<std::string, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> array_map = LoadMatlabFile("trainingData.mat", array_names);
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
        std::vector<ait::ImageSample> samples;
        for (auto i = 0; i < images.size(); i++)
        {
            for (int x=0; x < images[i].get_data_matrix().rows(); x++)
            {
                if (x % 4 != 0)
                    continue;
//                for (int y=0; y < images[i].get_data_matrix().cols(); y++)
                {
                int y = 0;
					ait::ImageSample sample(&images[i], x, y);
					samples.push_back(std::move(sample));
                }
            }
        }
        std::cout << " Done." << std::endl;

        using RandomEngine = std::mt19937_64;
        using SampleIteratorType = std::vector<ait::ImageSample>::iterator;
        using ConstSampleIteratorType = std::vector<ait::ImageSample>::const_iterator;
        using StatisticsFactoryType = typename ait::HistogramStatistics<ait::ImageSample>::HistogramStatisticsFactory;
        using WeakLearnerType = ait::ImageWeakLearner<StatisticsFactoryType, SampleIteratorType, RandomEngine>;
        using ForestType = ait::Forest<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample>>;
        
        StatisticsFactoryType statistics_factory(num_of_classes);
        ait::ImageWeakLearnerParameters weak_learner_parameters;
        ait::TrainingParameters training_parameters;
        WeakLearnerType iwl(weak_learner_parameters, statistics_factory);

        ait::ForestTrainer<ait::ImageSample, WeakLearnerType, RandomEngine> trainer(iwl, training_parameters);

		auto start_time = std::chrono::high_resolution_clock::now();
		ForestType forest = trainer.train_forest(samples);
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
            std::vector<std::vector<ait::size_type>> forest_leaf_indices = forest.evaluate(samples.cbegin(), samples.cend());

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
                    if (label == it->get_label())
                        match++;
                    else
                        no_match++;
                }
            }
            std::cout << "Match: " << match << ", no match: " << no_match << std::endl;

    //        forest.Evaluate(samples, [] (const typename ForestType::NodeType &node)
    //        {
    //            std::cout << "a" << std::endl;
    //        });

            ait::Tree<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample>> tree = forest.get_tree(0);
            ait::TreeUtilities<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample>, ConstSampleIteratorType> tree_utils(tree);
            auto matrix = tree_utils.compute_confusion_matrix<3>(samples.cbegin(), samples.cend());
            std::cout << "Confusion matrix:" << std::endl << matrix << std::endl;
            auto norm_matrix = tree_utils.compute_normalized_confusion_matrix<3>(samples.cbegin(), samples.cend());
            std::cout << "Normalized confusion matrix:" << std::endl << norm_matrix << std::endl;
        }
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}
