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
#if AIT_TESTING
    int num_of_trees = 1;
    int tree_depth = 12;
#else
    int num_of_trees = 1;
    int tree_depth = 20;
#endif
    int minimum_num_of_samples = 0;
    double minimum_information_gain = 0.0;
#if AIT_MULTI_THREADING
    int num_of_threads = -1;
#endif
};

struct LevelTrainingParameters : public TrainingParameters
{
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
