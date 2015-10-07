//
//  level_forest_trainer.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 30/09/15.
//
//

#pragma once

#include <iostream>
#include <sstream>
#include <map>

#include "ait.h"
#include "training.h"
#include "forest.h"
#include "weak_learner.h"

namespace ait
{

struct LevelTrainingParameters : public TrainingParameters
{
    // TODO
};

template <typename TSamplePointerIterator, typename TWeakLearner, typename TRandomEngine = std::mt19937_64>
class LevelForestTrainer
{
public:
    using SamplePointerT = typename TSamplePointerIterator::value_type;
    using StatisticsT = typename TWeakLearner::StatisticsT;
    using SampleIteratorT = typename TWeakLearner::SampleIteratorT;
    using SplitPointT = typename TWeakLearner::SplitPointT;
    using ForestType = Forest<SplitPointT, StatisticsT>;
    using TreeType = Tree<SplitPointT, StatisticsT>;
    using NodeType = typename TreeType::NodeType;
    using NodeIterator = typename TreeType::NodeIterator;

protected:
    const TWeakLearner weak_learner_;
    const LevelTrainingParameters training_parameters_;
    
    void output_spaces(std::ostream &stream, int num_of_spaces) const
    {
        for (int i = 0; i < num_of_spaces; i++)
            stream << " ";
    }
    
    std::vector<SplitPointT> sample_split_points_batch(size_type num_of_nodes, SampleIteratorT i_start, SampleIteratorT i_end, TRandomEngine rnd_engine)
    {
        std::vector<SplitPointT> split_points_batch;
        for (size_type i = 0; i < num_of_nodes; i++)
        {
            std::vector<SplitPointT> split_points = weak_learner_.samplesp_split_points(i_start, i_end, rnd_engine);
            split_points_batch.insert(split_points_batch.end(),
                                      std::make_move_iterator(split_points.begin()),
                                      std::make_move_iterator(split_points.end())
                                      );
        }
        return split_points_batch;
    }

    virtual std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > sample_split_points_batch(const std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map, TRandomEngine &rnd_engine) const
    {
        std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > split_points_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            typename TWeakLearner::SampleIteratorT sample_it_begin = make_pointer_iterator_wrapper(map_it->second.cbegin());
            typename TWeakLearner::SampleIteratorT sample_it_end = make_pointer_iterator_wrapper(map_it->second.cend());
            split_points_batch[map_it->first] = weak_learner_.sample_split_points(sample_it_begin, sample_it_end, rnd_engine);
        }
        return split_points_batch;
    }
    
    std::map<typename TreeType::NodeIterator, SplitStatistics<StatisticsT> > compute_split_statistics_batch(const std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map, const std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > &split_points_batch) const
    {
        std::map<typename TreeType::NodeIterator, SplitStatistics<StatisticsT> > split_statistics_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            typename TWeakLearner::SampleIteratorT sample_it_begin = make_pointer_iterator_wrapper(map_it->second.begin());
            typename TWeakLearner::SampleIteratorT sample_it_end = make_pointer_iterator_wrapper(map_it->second.end());
            const std::vector<SplitPointT> &split_points = split_points_batch.at(map_it->first);
            split_statistics_batch[map_it->first] = weak_learner_.compute_split_statistics(sample_it_begin, sample_it_end, split_points);
        }
        return split_statistics_batch;
    }
    
    std::map<typename TreeType::NodeIterator, SplitPointT> find_best_split_point_batch(const std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > &split_points_batch, const std::map<typename TreeType::NodeIterator, StatisticsT> &current_statistics, const std::map<typename TreeType::NodeIterator, SplitStatistics<StatisticsT> > &split_statistics_batch) const
    {
        std::map<typename TreeType::NodeIterator, SplitPointT> best_split_point_batch;
        for (auto map_it = split_statistics_batch.begin(); map_it != split_statistics_batch.end(); ++map_it)
        {
            const std::vector<SplitPointT> & split_points = split_points_batch.at(map_it->first);
            std::tuple<size_type, scalar_type> best_split_point_tuple = weak_learner_.find_best_split_point_tuple(current_statistics.at(map_it->first), map_it->second);
            size_type best_split_point_index = std::get<0>(best_split_point_tuple);
            SplitPointT best_split_point = split_points[best_split_point_index];
            best_split_point_batch[map_it->first] = best_split_point;

        }
        return best_split_point_batch;
    }

    std::map<typename TreeType::NodeIterator, StatisticsT> compute_statistics_batch(std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map) const
    {
        std::map<typename TreeType::NodeIterator, StatisticsT> statistics_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            typename TWeakLearner::SampleIteratorT sample_it_begin = make_pointer_iterator_wrapper(map_it->second.begin());
            typename TWeakLearner::SampleIteratorT sample_it_end = make_pointer_iterator_wrapper(map_it->second.end());
            typename TreeType::NodeIterator node_it = map_it->first;
            StatisticsT statistics = weak_learner_.create_statistics();
            statistics.accumulate(sample_it_begin, sample_it_end);
            statistics_batch.insert(std::make_pair(node_it, std::move(statistics)));
        }
        // Receive statistics from rank > 0
        //dist_statistics_batch = receive();
        std::map<typename TreeType::NodeIterator, std::vector<StatisticsT> > dist_statistics_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            dist_statistics_batch[map_it->first].push_back(statistics_batch.at(map_it->first));
        }
        for (auto map_it = dist_statistics_batch.begin(); map_it != dist_statistics_batch.end(); ++map_it)
        {
            auto it_start = map_it->second.begin();
            auto it_end = map_it->second.end();
            StatisticsT statistics = weak_learner_.create_statistics();
            statistics.accumulate_histograms(it_start, it_end);
            typename TreeType::NodeIterator node_it = map_it->first;
            node_it->set_statistics(statistics);
            statistics_batch.at(map_it->first) = std::move(statistics);
        }
        return statistics_batch;
    }

    std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > get_sample_node_map(TreeType &tree, TSamplePointerIterator samples_start, TSamplePointerIterator samples_end) const
    {
        std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > node_to_sample_map;
        for (auto it = samples_start; it != samples_end; ++it)
        {
            // TODO: Use template meta-programming to allow tree.evaluate to be called for both samples and sample-iterators.
            typename TreeType::NodeIterator node_it = tree.evaluate(*(*it));
            node_to_sample_map[node_it].push_back(*it);
        }
        return node_to_sample_map;
    }

    virtual void broadcast_tree(TreeType &tree) const
    {}

public:
    LevelForestTrainer(const TWeakLearner &weak_learner, const LevelTrainingParameters &training_parameters)
    : weak_learner_(weak_learner), training_parameters_(training_parameters)
    {}

    TreeType train_tree(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_tree(samples_start, samples_end, rnd_engine);
    }
    
    void train_tree_level(TreeType &tree, size_type current_level, TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, TRandomEngine &rnd_engine) const
    {
        std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > node_to_sample_map = get_sample_node_map(tree, samples_start, samples_end);
        std::cout << "current_level: " << current_level << ", # nodes: " << node_to_sample_map.size() << std::endl;
        const std::map<typename TreeType::NodeIterator, StatisticsT> &current_statistics = compute_statistics_batch(node_to_sample_map);
        if (current_level < training_parameters_.tree_depth)
        {
            std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > split_points_batch = sample_split_points_batch(node_to_sample_map, rnd_engine);
            std::map<typename TreeType::NodeIterator, SplitStatistics<StatisticsT> > split_statistics_batch = compute_split_statistics_batch(node_to_sample_map, split_points_batch);
            // Receive split statistics from rank > 0
            std::map<typename TreeType::NodeIterator, SplitPointT> best_split_point_batch = find_best_split_point_batch(split_points_batch, current_statistics, split_statistics_batch);
            for (auto map_it = best_split_point_batch.begin(); map_it != best_split_point_batch.end(); ++map_it)
            {
                typename TreeType::NodeIterator node_it = map_it->first;
                node_it->set_split_point(map_it->second);
                node_it.set_leaf(false);
                node_it.left_child().set_leaf(true);
                node_it.right_child().set_leaf(true);
            }
        }
    }

    TreeType train_tree(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, TRandomEngine &rnd_engine) const
    {
        TreeType tree(training_parameters_.tree_depth);
        tree.get_root_iterator().set_leaf();
        std::cout << "Training tree, # samples " << (samples_end - samples_start) << std::endl;
        for (size_type current_level = 1; current_level <= training_parameters_.tree_depth; current_level++)
        {
            train_tree_level(tree, current_level, samples_start, samples_end, rnd_engine);
            // save new tree
            broadcast_tree(tree);
        }
        return tree;
    }

    ForestType train_forest(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, TRandomEngine &rnd_engine) const
    {
        ForestType forest;
        for (int i=0; i < training_parameters_.num_of_trees; i++)
        {
            TreeType tree = train_tree(samples_start, samples_end, rnd_engine);
            forest.add_tree(std::move(tree));
        }
        return forest;
    }

    ForestType train_forest(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_forest(samples_start, samples_end, rnd_engine);
    }

};
    
}

