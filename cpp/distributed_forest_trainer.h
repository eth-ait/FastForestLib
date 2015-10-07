//
//  distributed_forest_trainer.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 30/09/15.
//
//

#pragma once

#include <iostream>
#include <sstream>
#include <map>

#include <boost/serialization/map.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

#include "ait.h"
#include "level_forest_trainer.h"

namespace ait
{
    
namespace mpi = boost::mpi;

struct DistributedTrainingParameters : public LevelTrainingParameters
{
    // TODO
};

template <typename TSamplePointerIterator, typename TWeakLearner, typename TRandomEngine = std::mt19937_64>
class DistributedForestTrainer : public LevelForestTrainer<TSamplePointerIterator, TWeakLearner, TRandomEngine>
{
    using BaseType = LevelForestTrainer<TSamplePointerIterator, TWeakLearner, TRandomEngine>;
    
    using SamplePointerT = typename BaseType::SamplePointerT;
    using StatisticsT = typename BaseType::StatisticsT;
    using SampleIteratorT = typename BaseType::SampleIteratorT;
    using SplitPointT = typename BaseType::SplitPointT;
    using ForestType = typename BaseType::ForestType;
    using TreeType = typename BaseType::TreeType;
    using NodeType = typename TreeType::NodeType;
    using NodeIterator = typename TreeType::NodeIterator;

public:
    DistributedForestTrainer(mpi::communicator &comm, const TWeakLearner &weak_learner, const DistributedTrainingParameters &training_parameters)
    : BaseType(weak_learner, training_parameters), comm_(comm)
    {}

protected:
    virtual void broadcast_tree(TreeType &tree) const override
    {
        TreeType bcast_tree;
        if (comm_.rank() == 0)
        {
            bcast_tree = std::move(tree);
        }
        broadcast(comm_, bcast_tree, 0);
        tree = std::move(bcast_tree);
    }

    virtual std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > sample_split_points_batch(const std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map, TRandomEngine &rnd_engine) const override
    {
        std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > split_points_batch;
        if (comm_.rank() == 0)
        {
            split_points_batch = BaseType::sample_split_points_batch(node_to_sample_map, rnd_engine);
        }
        broadcast(comm_, split_points_batch, 0);
        return split_points_batch;
    }

//    void train_tree_level(TreeType &tree, size_type current_level, TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, TRandomEngine &rnd_engine) const
//    {
//        std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > node_to_sample_map = get_sample_node_map(tree, samples_start, samples_end);
//        std::cout << "current_level: " << current_level << ", # nodes: " << node_to_sample_map.size() << std::endl;
//        const std::map<typename TreeType::NodeIterator, StatisticsT> &current_statistics = compute_statistics_batch(node_to_sample_map);
//        if (current_level < training_parameters_.tree_depth)
//        {
//            std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > split_points_batch = sample_split_points_batch(node_to_sample_map, rnd_engine);
//            // Distribute split points
//            std::map<typename TreeType::NodeIterator, SplitStatistics<StatisticsT> > split_statistics_batch = compute_split_statistics_batch(node_to_sample_map, split_points_batch);
//            // Receive split statistics from rank > 0
//            std::map<typename TreeType::NodeIterator, SplitPointT> best_split_point_batch = find_best_split_point_batch(split_points_batch, current_statistics, split_statistics_batch);
//            for (auto map_it = best_split_point_batch.begin(); map_it != best_split_point_batch.end(); ++map_it)
//            {
//                typename TreeType::NodeIterator node_it = map_it->first;
//                node_it->set_split_point(map_it->second);
//                node_it.set_leaf(false);
//                node_it.left_child().set_leaf(true);
//                node_it.right_child().set_leaf(true);
//            }
//        }
//    }

private:
    mpi::communicator comm_;
};
    
}
