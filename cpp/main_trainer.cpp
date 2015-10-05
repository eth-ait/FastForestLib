#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <random>
#include <iostream>
#include <chrono>

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>

#include "ait.h"
#include "forest_trainer.h"
#include "histogram_statistics.h"
#include "image_weak_learner.h"
#include "eigen_matrix_io.h"
#include "matlab_file_io.h"


int main(int argc, const char *argv[]) {
    try {
        //		std::vector<std::string> array_names;
        //		array_names.push_back("data");
        //		array_names.push_back("labels");
        //		std::map<std::string, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> array_map = LoadMatlabFile("trainingData.mat", array_names);
        std::cout << "Reading images... " << std::flush;
        std::vector<ait::Image> images = ait::load_images_from_matlab_file("../../data/trainingData.mat", "data", "labels");
        std::cout << "Done." << std::endl;
        
        std::cout << "Creating samples... " << std::flush;
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
        std::cout << "Done." << std::endl;

        using RandomEngine = std::mt19937_64;
        using SampleIteratorType = std::vector<ait::ImageSample>::iterator;
        using ConstSampleIteratorType = std::vector<ait::ImageSample>::const_iterator;
        using WeakLearnerType = ait::ImageWeakLearner<ait::HistogramStatistics<ait::ImageSample>, SampleIteratorType, RandomEngine>;
        using ForestType = ait::Forest<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample> >;

        ait::ImageWeakLearnerParameters weak_learner_parameters;
        ait::TrainingParameters training_parameters;
        WeakLearnerType iwl(weak_learner_parameters);

        ait::ForestTrainer<ait::ImageSample, WeakLearnerType, RandomEngine> trainer(iwl, training_parameters);

		auto start_time = std::chrono::high_resolution_clock::now();
		ForestType forest = trainer.train_forest(samples);
		auto stop_time = std::chrono::high_resolution_clock::now();

		auto duration = stop_time - start_time;
		auto period = std::chrono::high_resolution_clock::period();
		double elapsed_seconds = duration.count() * period.num / static_cast<double>(period.den);
		std::cout << "Running time: " << elapsed_seconds << std::endl;

        // Serialize forest
        {
            std::cout << "Writing json forest file ... " << std::flush;
            std::ofstream ofile("forest.json");
            cereal::JSONOutputArchive oarchive(ofile);
            oarchive(cereal::make_nvp("forest", forest));
            std::cout << "done." << std::endl;
        }

        {
            std::cout << "Writing binary forest file ... " << std::flush;
            std::ofstream bin_ofile("forest.bin", std::ios::binary);
            cereal::BinaryOutputArchive bin_oarchive(bin_ofile);
            bin_oarchive(cereal::make_nvp("forest", forest));
            std::cout << "done." << std::endl;
        }

        {
            std::cout << "Reading json forest file ... " << std::flush;
            std::ifstream ifile("forest.json");
            cereal::JSONInputArchive iarchive(ifile);
            iarchive(forest);
            std::cout << "done." << std::endl;
        }

        {
            std::cout << "Reading binary forest file ... " << std::flush;
            std::ifstream bin_ifile("forest.bin", std::ios::binary);
            cereal::BinaryInputArchive bin_iarchive(bin_ifile);
            bin_iarchive(forest);
            std::cout << "done." << std::endl;
        }

        std::vector<std::vector<ait::size_type> > forest_leaf_indices = forest.evaluate(samples.cbegin(), samples.cend());

        int match = 0;
        int no_match = 0;
        for (std::size_t i=0; i < forest.size(); i++)
        {
            const ForestType::TreeType &tree = forest.get_tree(i);
            for (auto it=samples.cbegin(); it != samples.cend(); it++)
            {
                const auto &node = *tree.get_node(forest_leaf_indices[i][it - samples.cbegin()]);
                const auto &statistics = node.get_statistics();
                auto max_it = std::max_element(statistics.get_histogram().cbegin(), statistics.get_histogram().cend());
                auto label = max_it - statistics.get_histogram().cbegin();
                if (label == it->get_label())
                    match++;
                else
                    no_match++;
            }
        }
        std::cout << "match: " << match << ", no_match: " << no_match << std::endl;

//        forest.Evaluate(samples, [] (const typename ForestType::NodeType &node)
//        {
//            std::cout << "a" << std::endl;
//        });

        ait::Tree<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample> > tree = forest.get_tree(0);
        ait::TreeUtilities<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample>, ConstSampleIteratorType> tree_utils(tree);
        auto matrix = tree_utils.compute_confusion_matrix<3>(samples.cbegin(), samples.cend());
        std::cout << "Confusion matrix:" << std::endl << matrix << std::endl;
        auto norm_matrix = tree_utils.compute_normalized_confusion_matrix<3>(samples.cbegin(), samples.cend());
        std::cout << "Normalized confusion matrix:" << std::endl << norm_matrix << std::endl;
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}
