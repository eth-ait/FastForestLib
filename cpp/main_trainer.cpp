#include <strstream>
#include <vector>
#include <map>
#include <random>
#include <iostream>

#include "histogram_statistics.h"
#include "image_weak_learner.h"
#include "eigen_matrix_io.h"
#include "matlab_file_io.h"
#include "forest_trainer.h"


int main(int argc, const char *argv[]) {
    try {
        //		std::vector<std::string> array_names;
        //		array_names.push_back("data");
        //		array_names.push_back("labels");
        //		std::map<std::string, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> array_map = LoadMatlabFile("trainingData.mat", array_names);
        std::cout << "Reading images... " << std::flush;
        std::vector<AIT::Image<> > images = AIT::LoadImagesFromMatlabFile("../../data/trainingData.mat", "data", "labels");
        std::cout << "Done." << std::endl;
        
        std::cout << "Creating samples... " << std::flush;
        std::vector<AIT::ImageSample<> > samples;
        for (auto i = 0; i < images.size(); i++) {
            for (int x=0; x < images[i].GetDataMatrix().rows(); x++) {
                int y = 0;
                AIT::ImageSample<> sample(&images[i], x, y);
                samples.push_back(std::move(sample));
            }
        }
        std::cout << "Done." << std::endl;
        
        typedef std::mt19937_64 RandomEngine;
        typedef AIT::ImageSample<> SampleType;
        typedef std::vector<AIT::ImageSample<> >::iterator SampleIteratorType;
        
        AIT::ImageWeakLearnerParameters weak_learner_parameters;
        typedef AIT::ImageWeakLearner<AIT::HistogramStatistics<AIT::ImageSample<> >, SampleIteratorType, RandomEngine> WeakLearnerType;
        WeakLearnerType iwl(weak_learner_parameters);
        AIT::TrainingParameters training_parameters;
        
        AIT::ForestTrainer<AIT::ImageSample<>, WeakLearnerType, AIT::TrainingParameters, RandomEngine> trainer(iwl, training_parameters);
        typedef AIT::Forest<AIT::ImageSplitPoint<>, AIT::HistogramStatistics<AIT::ImageSample<> > > ForestType;
        ForestType forest = trainer.TrainForest(samples);
        
        std::vector<std::vector<std::size_t> > forest_leaf_indices = forest.Evaluate<AIT::ImageSample<> >(samples);
        
        
        int match = 0;
        int no_match = 0;
        for (std::size_t i=0; i < forest.NumOfTrees(); i++) {
            const ForestType::TreeType &tree = forest.GetTree(i);
            for (auto it=samples.cbegin(); it != samples.cend(); it++) {
                const auto &node = *tree.GetNode(forest_leaf_indices[i][it - samples.cbegin()]);
                const auto &statistics = node.GetStatistics();
                auto max_it = std::max_element(statistics.GetHistogram().cbegin(), statistics.GetHistogram().cend());
                auto label = max_it - statistics.GetHistogram().cbegin();
                if (label == it->GetLabel())
                    match++;
                else
                    no_match++;
            }
        }
        std::cout << "match: " << match << ", no_match: " << no_match << std::endl;
        
        //        forest.Evaluate(samples, [] (const typename ForestType::NodeType &node) {
        //            std::cout << "a" << std::endl;
        //        });
    }
    catch (const std::runtime_error &error) {
        std::cerr << "Runtime exception occured" << std::endl;
        std::cerr << error.what() << std::endl;
    }
    
    return 0;
}
