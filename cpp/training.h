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
#include "config_utils.h"

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
	int_type tree_depth = 20;
#endif

    // If a node contains less samples than minimum_num_of_samples it is not split anymore
    int_type minimum_num_of_samples = 100;
    // Minimum information gain to achieve before stopping splitting of nodes
    double minimum_information_gain = 0.0;

#if AIT_MULTI_THREADING
    // Number of threads to use for statistics computation
    int_type num_of_threads = -1;
#endif

    virtual void read_from_config(const rapidjson::Value& config) {
        num_of_trees = ConfigurationUtils::get_int_value(config, "num_of_trees", num_of_trees);
        tree_depth = ConfigurationUtils::get_int_value(config, "tree_depth", tree_depth);
        minimum_num_of_samples = ConfigurationUtils::get_int_value(config, "minimum_num_of_samples", minimum_num_of_samples);
        minimum_information_gain = ConfigurationUtils::get_double_value(config, "minimum_information_gain", minimum_information_gain);
        num_of_threads = ConfigurationUtils::get_int_value(config, "num_of_threads", num_of_threads);
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
    
    virtual void read_from_config(const rapidjson::Value& config) {
        TrainingParameters::read_from_config(config);
        level_part_size = ConfigurationUtils::get_int_value(config, "level_part_size", level_part_size);
        temporary_json_forest_file_prefix = ConfigurationUtils::get_string_value(config, "temporary_json_forest_file_prefix", temporary_json_forest_file_prefix);
        temporary_binary_forest_file_prefix = ConfigurationUtils::get_string_value(config, "temporary_binary_forest_file_prefix", temporary_binary_forest_file_prefix);
        temporary_json_tree_file_prefix = ConfigurationUtils::get_string_value(config, "temporary_json_tree_file_prefix", temporary_json_tree_file_prefix);
        temporary_binary_tree_file_prefix = ConfigurationUtils::get_string_value(config, "temporary_binary_tree_file_prefix", temporary_binary_tree_file_prefix);
    }
};

// TODO: Is this ever needed?
struct DistributedTrainingParameters : public LevelTrainingParameters {
    virtual void read_from_config(const rapidjson::Value& config) {
        LevelTrainingParameters::read_from_config(config);
    }
};

}
