//
//  training.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 06/10/15.
//
//

#pragma once

#include "ait.h"
#include "mpl_utils.h"

namespace ait
{

struct TrainingParameters
{
    // Number of trees to train and their depth
#if AIT_TESTING
    int_type num_of_trees = 1;
	int_type tree_depth = 12;
#else
	int_type num_of_trees = 3;
	int_type tree_depth = 21;
#endif

    // If a node contains less samples than minimum_num_of_samples it is not split anymore
    int_type minimum_num_of_samples = 100;
    // Minimum information gain to achieve before stopping splitting of nodes
    double minimum_information_gain = 0.0;

#if AIT_MULTI_THREADING
    // Number of threads to use for statistics computation
    int_type num_of_threads = -1;
#endif

private:
    friend class cereal::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("num_of_trees", num_of_trees));
        archive(cereal::make_nvp("tree_depth", tree_depth));
        archive(cereal::make_nvp("minimum_num_of_samples", minimum_num_of_samples));
        archive(cereal::make_nvp("minimum_information_gain", minimum_information_gain));
        archive(cereal::make_nvp("num_of_threads", num_of_threads));
    }
};

struct LevelTrainingParameters : public TrainingParameters
{
    // Number of nodes that are trained in one batch (otherwise memory will grow very quickly with deeper levels)
    int_type level_part_size = 256;
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
