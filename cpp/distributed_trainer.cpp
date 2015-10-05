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

#include "ait.h"
#include "distributed_forest_trainer.h"
#include "image_weak_learner.h"
#include "matlab_file_io.h"

int main(int argc, const char *argv[]) {
    try {
        std::cout << "Reading images... " << std::flush;
        std::vector<ait::Image> images = ait::load_images_from_matlab_file("../../data/trainingData.mat", "data", "labels");
        std::cout << "Done." << std::endl;
        
        std::cout << "Creating samples... " << std::flush;
        using ImageSamplePointer = std::shared_ptr<ait::ImageSample>;
        std::vector<ImageSamplePointer> samples;
        for (auto i = 0; i < images.size(); i++) {
            for (int x=0; x < images[i].get_data_matrix().rows(); x++)
            {
//                if (x % 4 != 0)
//                    continue;
                for (int y=0; y < images[i].get_data_matrix().cols(); y++)
                {
//                    int y = 0;
                    ImageSamplePointer sample_ptr = std::make_shared<ait::ImageSample>(&images[i], x, y);
                    samples.push_back(sample_ptr);
                }
            }
        }
        std::cout << "Done." << std::endl;

        using SamplePointerIteratorType = std::vector<ImageSamplePointer>::const_iterator;
        using SampleIteratorType = ait::PointerIteratorWrapper<SamplePointerIteratorType>;
        using WeakLearnerType = ait::ImageWeakLearner<ait::HistogramStatistics<ait::ImageSample>, SampleIteratorType>;
        using RandomEngine = std::mt19937_64;

        ait::ImageWeakLearnerParameters weak_learner_parameters;
        ait::DistributedTrainingParameters training_parameters;
        WeakLearnerType iwl(weak_learner_parameters);
        
        ait::DistributedForestTrainer<SamplePointerIteratorType, WeakLearnerType, RandomEngine> trainer(iwl, training_parameters);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        using ForestType = ait::Forest<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample> >;
        ForestType forest = trainer.train_forest(samples.cbegin(), samples.cend());
        auto stop_time = std::chrono::high_resolution_clock::now();
        auto duration = stop_time - start_time;
        auto period = std::chrono::high_resolution_clock::period();
        double elapsed_seconds = duration.count() * period.num / static_cast<double>(period.den);
        std::cout << "Running time: " << elapsed_seconds << std::endl;

        // Serialize forest to file
        {
            std::cout << "Writing json forest file ... " << std::flush;
            std::ofstream ofile("forest.json");
            cereal::JSONOutputArchive oarchive(ofile);
            oarchive(cereal::make_nvp("forest", forest));
            std::cout << "done." << std::endl;
        }
        // Read forest from file for testing
        {
            std::cout << "Reading json forest file ... " << std::flush;
            std::ifstream ifile("forest.json");
            cereal::JSONInputArchive iarchive(ifile);
            iarchive(forest);
            std::cout << "done." << std::endl;
        }

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
                const auto &node = *tree.get_node(forest_leaf_indices[i][it - samples.cbegin()]);
                const auto &statistics = node.get_statistics();
                auto max_it = std::max_element(statistics.get_histogram().cbegin(), statistics.get_histogram().cend());
                auto label = max_it - statistics.get_histogram().cbegin();
                if (label == (*it)->get_label())
                    match++;
                else
                    no_match++;
            }
        }
        std::cout << "match: " << match << ", no_match: " << no_match << std::endl;

        ait::Tree<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample> > tree = forest.get_tree(0);
        ait::TreeUtilities<ait::ImageSplitPoint, ait::HistogramStatistics<ait::ImageSample>, SampleIteratorType> tree_utils(tree);
        auto matrix = tree_utils.compute_confusion_matrix<3>(samples_start, samples_end);
        std::cout << "Confusion matrix:" << std::endl << matrix << std::endl;
        auto norm_matrix = tree_utils.compute_normalized_confusion_matrix<3>(samples_start, samples_end);
        std::cout << "Normalized confusion matrix:" << std::endl << norm_matrix << std::endl;
    }
    catch (const std::runtime_error &error)
    {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}
