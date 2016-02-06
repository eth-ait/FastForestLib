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

#ifdef SERIALIZE_WITH_BOOST
#include <boost/serialization/vector.hpp>
#include "serialization_utils.h"
#endif

#include "ait.h"
#include "level_forest_trainer.h"

namespace ait
{

namespace mpi = boost::mpi;

template <template <typename> class TWeakLearner, typename TSampleIterator>
class DistributedForestTrainer : public LevelForestTrainer<TWeakLearner, TSampleIterator>
{
    using BaseT = LevelForestTrainer<TWeakLearner, TSampleIterator>;

public:
    struct DistributedTrainingParameters : public BaseT::ParametersT
    {
        // TODO: Add distributed training parameters?
    };

    using ParametersT = DistributedTrainingParameters;
    
    using SamplePointerT = typename BaseT::SamplePointerT;
    using WeakLearnerT = typename BaseT::WeakLearnerT;
    using RandomEngineT = typename BaseT::RandomEngineT;

    using SplitPointT = typename BaseT::SplitPointT;
    using SplitPointCandidatesT = typename BaseT::SplitPointCandidatesT;
    using StatisticsT = typename BaseT::StatisticsT;
    using TreeT = typename BaseT::TreeT;

    explicit DistributedForestTrainer(mpi::communicator& comm, const WeakLearnerT& weak_learner, const ParametersT& training_parameters)
    : BaseT(weak_learner, training_parameters), comm_(comm), training_parameters_(training_parameters)
    {}
    
    virtual ~DistributedForestTrainer()
    {}

protected:
    template <typename T> using TreeNodeMap = typename BaseT::template TreeNodeMap<T>;

    void broadcast_tree(TreeT& tree, int root = 0) const
    {
#if AIT_PROFILE
        auto start_time = std::chrono::high_resolution_clock::now();
        if (comm_.rank() == root)
        {
            log_profile(false) << "Broadcasting tree ...";
        }
#endif
        if (comm_.rank() == root)
        {
            broadcast(comm_, tree, root);
        }
        else
        {
            TreeT bcast_tree;
            broadcast(comm_, bcast_tree, root);
            tree = std::move(bcast_tree);
        }
#if AIT_PROFILE
        if (comm_.rank() == root)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
        }
#endif
    }

    template <typename ValueType>
    void broadcast_tree_node_map(TreeT& tree, TreeNodeMap<ValueType>& map, int root = 0) const
    {
        std::map<size_type, ValueType> wrapper_map;
        if (comm_.rank() != root)
        {
            map.clear();
        }
        broadcast(comm_, map.base_map(), root);
    }
    
    void exchange_split_statistics_batch(TreeT& tree, TreeNodeMap<SplitStatistics<StatisticsT>>& map, int root = 0) const
    {
#if AIT_PROFILE
        auto start_time = std::chrono::high_resolution_clock::now();
        if (comm_.rank() == root)
        {
            log_profile(false) << "Exchanging split statistics batch ...";
        }
#endif
        std::vector<TreeNodeMap<SplitStatistics<StatisticsT>>> maps(comm_.size(), TreeNodeMap<SplitStatistics<StatisticsT>>(tree));
        gather(comm_, map, &maps[0], root);
#if AIT_PROFILE
        if (comm_.rank() == root)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
            start_time = std::chrono::high_resolution_clock::now();
            log_profile(false) << "Accumulating statistics ...";
        }
#endif
        if (comm_.rank() == root)
        {
            log_debug() << "First split_statistics size: " << map.cbegin()->get_left_statistics(0).num_of_samples();
            map.clear();
            for (auto it = maps.cbegin(); it != maps.cend(); ++it)
            {
                for (auto map_it = it->cbegin(); map_it != it->cend(); ++map_it)
                {
                    SplitStatistics<StatisticsT>& split_statistics = map[map_it.node_iterator()];
                    const SplitStatistics<StatisticsT>& other_split_statistics = *map_it;
                    if (split_statistics.size() == 0)
                    {
                        split_statistics = other_split_statistics;
                    }
                    else
                    {
                        split_statistics.accumulate(other_split_statistics);
                    }
                }
            }
            log_debug() << "After accumulation: " << map.cbegin()->get_left_statistics(0).num_of_samples();
        }
#if AIT_PROFILE
        if (comm_.rank() == root)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
            start_time = std::chrono::high_resolution_clock::now();
            log_profile(false) << "Broadcasting tree node map ...";
        }
#endif
        broadcast_tree_node_map(tree, map, root);
#if AIT_PROFILE
        if (comm_.rank() == root)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
        }
#endif
    }

    virtual TreeNodeMap<SplitStatistics<StatisticsT>> compute_split_statistics_batch(TreeT& tree, const TreeNodeMap<std::vector<SamplePointerT>>& node_to_sample_map, const TreeNodeMap<SplitPointCandidatesT>& split_points_batch) const override
    {
#if AIT_PROFILE
        auto start_time = std::chrono::high_resolution_clock::now();
        if (comm_.rank() == 0)
        {
            log_profile(false) << "Computing split statistics batch ...";
        }
#endif
        TreeNodeMap<SplitStatistics<StatisticsT>> split_statistics_batch = BaseT::compute_split_statistics_batch(tree, node_to_sample_map, split_points_batch);
#if AIT_PROFILE
        if (comm_.rank() == 0)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
        }
#endif
        exchange_split_statistics_batch(tree, split_statistics_batch);
        return split_statistics_batch;
    }

    virtual TreeNodeMap<SplitPointCandidatesT> sample_split_points_batch(TreeT& tree, const TreeNodeMap<std::vector<SamplePointerT>>& node_to_sample_map, RandomEngineT& rnd_engine) const override
    {
#if AIT_PROFILE
        auto start_time = std::chrono::high_resolution_clock::now();
        if (comm_.rank() == 0)
        {
            log_profile(false) << "Sampling split points batch ...";
        }
#endif
        TreeNodeMap<SplitPointCandidatesT> split_points_batch(tree);
        if (comm_.rank() == 0)
        {
            split_points_batch = std::move(BaseT::sample_split_points_batch(tree, node_to_sample_map, rnd_engine));
        }
#if AIT_PROFILE
        if (comm_.rank() == 0)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
            start_time = std::chrono::high_resolution_clock::now();
            log_profile(false) << "Broadcasting split points batch ...";
        }
#endif
        broadcast_tree_node_map(tree, split_points_batch);
#if AIT_PROFILE
        if (comm_.rank() == 0)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
        }
#endif
        return split_points_batch;
    }
    
    void exchange_statistics_batch(TreeT& tree, TreeNodeMap<StatisticsT>& map, int root = 0) const
    {
#if AIT_PROFILE
        auto start_time = std::chrono::high_resolution_clock::now();
        if (comm_.rank() == root)
        {
            log_profile(false) << "Accumulating statistics batch ...";
        }
#endif
        std::vector<TreeNodeMap<StatisticsT>> maps(comm_.size(), TreeNodeMap<StatisticsT>(tree));
        gather(comm_, map, &maps[0], root);
        if (comm_.rank() == root)
        {
            log_debug() << "First statistics size: " << map.cbegin()->num_of_samples();
            map.clear();
            for (auto it = maps.cbegin(); it != maps.cend(); ++it)
            {
                log_debug() << "Processing statistics from rank " << (it - maps.cbegin());
                for (auto map_it = it->cbegin(); map_it != it->cend(); ++map_it)
                {
                    auto it = map.find(map_it.node_iterator());
                    if (it == map.end())
                    {
                        map[map_it.node_iterator()] = this->weak_learner_.create_statistics();
                    }
                    StatisticsT &statistics = map[map_it.node_iterator()];
//                    log_debug() << "Processing statistics for node " << map_it.node_iterator().get_node_index();
                    const StatisticsT &other_statistics = *map_it;
                    statistics.accumulate(other_statistics);
                }
            }
        }
#if AIT_PROFILE
        if (comm_.rank() == root)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
            start_time = std::chrono::high_resolution_clock::now();
            log_profile(false) << "Broadcasting statistics batch ...";
        }
#endif
        broadcast_tree_node_map(tree, map, root);
#if AIT_PROFILE
        if (comm_.rank() == root)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
        }
#endif
        // For debugging only:
//        if (comm_.rank() == root)
//        {
//            for (auto map_it = map.cbegin(); map_it != map.cend(); ++map_it)
//            {
//                auto logger = log_debug();
//                logger << "Num of samples for node [" << map_it.node_iterator().get_node_index() << "]: " << map_it->num_of_samples() << "{";
//                for (int i = 0; i < map_it->get_histogram().size(); ++i)
//                {
//                    logger << map_it->get_histogram()[i];
//                    if (i < map_it->get_histogram().size() - 1)
//                        logger << ", ";
//                }
//                logger << "}";
//            }
//        }
    }

    virtual TreeNodeMap<StatisticsT> compute_statistics_batch(TreeT &tree, TreeNodeMap<std::vector<SamplePointerT>> &node_to_sample_map) const override
    {
#if AIT_PROFILE
        auto start_time = std::chrono::high_resolution_clock::now();
        if (comm_.rank() == 0)
        {
            log_profile(false) << "Computing statistics batch ...";
        }
#endif
        TreeNodeMap<StatisticsT> statistics_batch = BaseT::compute_statistics_batch(tree, node_to_sample_map);
#if AIT_PROFILE
        if (comm_.rank() == 0)
        {
            ait::log_profile() << "Finished in " << compute_elapsed_milliseconds(start_time) << " ms";
        }
#endif
        exchange_statistics_batch(tree, statistics_batch);
        return statistics_batch;
    }

    virtual void train_tree_level(TreeT &tree, size_type current_level, TSampleIterator samples_start, TSampleIterator samples_end, RandomEngineT &rnd_engine) const override
    {
        BaseT::train_tree_level(tree, current_level, samples_start, samples_end, rnd_engine);
        broadcast_tree(tree);
    }

private:
    mpi::communicator comm_;
    ParametersT training_parameters_;
};
    
}
