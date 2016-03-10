//
//  training.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 06/10/15.
//
//

#pragma once

namespace ait
{

struct TrainingParameters
{
    // Number of trees to train and their depth
#if AIT_TESTING
    int num_of_trees = 1;
    int tree_depth = 12;
#else
    int num_of_trees = 1;
    int tree_depth = 12;
#endif
    // If a node contains less samples than minimum_num_of_samples it is not split anymore
    int minimum_num_of_samples = 100;
    // Minimum information gain to achieve before stopping splitting of nodes
    double minimum_information_gain = 0.0;
#if AIT_MULTI_THREADING
    // Number of threads to use for statistics computation
    int num_of_threads = -1;
#endif
};

struct LevelTrainingParameters : public TrainingParameters
{
    // Number of nodes that are trained in one batch (otherwise memory will grow very quickly with deeper levels)
    int level_part_size = 256;
    std::string temporary_json_forest_file_prefix;
    std::string temporary_binary_forest_file_prefix;
    std::string temporary_json_tree_file_prefix;
    std::string temporary_binary_tree_file_prefix;
};

// TODO: Is this ever needed?
struct DistributedTrainingParameters : public LevelTrainingParameters
{
};

}
