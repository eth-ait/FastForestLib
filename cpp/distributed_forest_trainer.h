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
    
template <template <typename, typename> class TWeakLearner, typename TSampleIterator, typename TRandomEngine = std::mt19937_64>
class DistributedForestTrainer : public LevelForestTrainer<TWeakLearner, TSampleIterator, TRandomEngine>
{
    using BaseT = LevelForestTrainer<TWeakLearner, TSampleIterator, TRandomEngine>;

public:
    struct DistributedTrainingParameters : public BaseT::ParametersT
    {
        // TODO
    };

    using ParametersT = DistributedTrainingParameters;
    
    using SamplePointerT = typename BaseT::SamplePointerT;
    using WeakLearnerT = typename BaseT::WeakLearnerT;
    using SplitPointT = typename BaseT::SplitPointT;
    using StatisticsT = typename BaseT::StatisticsT;
    using TreeT = typename BaseT::TreeT;

    DistributedForestTrainer(mpi::communicator &comm, const WeakLearnerT &weak_learner, const ParametersT &training_parameters)
    : BaseT(weak_learner, training_parameters), comm_(comm), training_parameters_(training_parameters)
    {}

protected:
    void exchange_tree(TreeT &tree) const
    {
        TreeT bcast_tree;
        if (comm_.rank() == 0)
        {
            bcast_tree = std::move(tree);
        }
        broadcast(comm_, bcast_tree, 0);
        tree = std::move(bcast_tree);
    }

    // TODO: Check that this is working
    void exchange_split_statistics_batch(const TreeT &tree, std::map<typename TreeT::NodeIterator, SplitStatistics<StatisticsT>> &map) const
    {
        std::map<size_type, SplitStatistics<StatisticsT>> wrapper_map;
        for (auto map_it = map.cbegin(); map_it != map.cend(); ++map_it)
        {
            size_type key_index = map_it->first - tree.cbegin();
            wrapper_map.insert(std::make_pair(key_index, map_it->second));
        }
        std::vector<std::map<size_type, SplitStatistics<StatisticsT>>> wrapper_maps;
        gather(comm_, wrapper_map, wrapper_maps, 0);
        if (comm_.rank() == 0)
        {
            typename TreeT::NodeIterator node_begin = map.begin()->first - (map.begin()->first - tree.cbegin());
            map.clear();
            for (auto it = wrapper_maps.cbegin(); it != wrapper_maps.cend(); ++it)
            {
                for (auto map_it = it->cbegin(); map_it != it->cend(); ++map_it)
                {
                    size_type key_index = map_it->first;
                    SplitStatistics<StatisticsT> &split_statistics = map[node_begin + key_index];
                    const SplitStatistics<StatisticsT> &wrapped_split_statistics = map_it->second;
                    split_statistics.accumulate(wrapped_split_statistics);
                }
            }
        }
        broadcast_map_with_tree_iterators(tree, map);
    }

    template <typename ValueType>
    void broadcast_map_with_tree_iterators(const TreeT &tree, std::map<typename TreeT::NodeIterator, ValueType> &map) const
    {
        std::map<size_type, ValueType> wrapper_map;
        if (comm_.rank() == 0)
        {
            for (auto map_it = map.cbegin(); map_it != map.cend(); ++map_it)
            {
                size_type key_index = map_it->first - tree.cbegin();
                wrapper_map.insert(std::make_pair(key_index, map_it->second));
            }
        }
        broadcast(comm_, wrapper_map, 0);
        if (comm_.rank() != 0)
        {
            typename TreeT::NodeIterator node_begin = map.begin()->first - (map.begin()->first - tree.cbegin());
            map.clear();
            for (auto map_it = wrapper_map.cbegin(); map_it != wrapper_map.cend(); ++map_it)
            {
                size_type key_index = map_it->first;
                map.insert(std::make_pair(node_begin + key_index, map_it->second));
            }
        }
    }
    
    virtual std::map<typename TreeT::NodeIterator, SplitStatistics<StatisticsT> > compute_split_statistics_batch(const TreeT &tree, const std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map, const std::map<typename TreeT::NodeIterator, std::vector<SplitPointT> > &split_points_batch) const override
    {
        std::map<typename TreeT::NodeIterator, SplitStatistics<StatisticsT> > split_statistics_batch;
        split_statistics_batch = BaseT::compute_split_statistics_batch(tree, node_to_sample_map, split_points_batch);
        exchange_split_statistics_batch(tree, split_statistics_batch);
        return split_statistics_batch;
    }
    

    virtual std::map<typename TreeT::NodeIterator, std::vector<SplitPointT>> sample_split_points_batch(const TreeT &tree, const std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT>> &node_to_sample_map, TRandomEngine &rnd_engine) const override
    {
        std::map<typename TreeT::NodeIterator, std::vector<SplitPointT>> split_points_batch;
        if (comm_.rank() == 0)
        {
            split_points_batch = BaseT::sample_split_points_batch(tree, node_to_sample_map, rnd_engine);
        }
        broadcast_map_with_tree_iterators(tree, split_points_batch);
        return split_points_batch;
    }

    virtual void train_tree_level(TreeT &tree, size_type current_level, TSampleIterator samples_start, TSampleIterator samples_end, TRandomEngine &rnd_engine) const override
    {
        BaseT::train_tree_level(tree, current_level, samples_start, samples_end, rnd_engine);
        exchange_tree(tree);
    }
//    void train_tree_level(TreeType &tree, size_type current_level, TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, TRandomEngine &rnd_engine) const
//    {
//        std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT>> node_to_sample_map = get_sample_node_map(tree, samples_start, samples_end);
//        std::cout << "current_level: " << current_level << ", # nodes: " << node_to_sample_map.size() << std::endl;
//        const std::map<typename TreeType::NodeIterator, StatisticsT> &current_statistics = compute_statistics_batch(node_to_sample_map);
//        if (current_level < training_parameters_.tree_depth)
//        {
//            std::map<typename TreeType::NodeIterator, std::vector<SplitPointT>> split_points_batch = sample_split_points_batch(node_to_sample_map, rnd_engine);
//            // Distribute split points
//            std::map<typename TreeType::NodeIterator, SplitStatistics<StatisticsT>> split_statistics_batch = compute_split_statistics_batch(node_to_sample_map, split_points_batch);
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
    ParametersT training_parameters_;
};
    
}
