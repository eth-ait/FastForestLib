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
#include "iterator_utils.h"
#include "training.h"
#include "forest.h"
#include "weak_learner.h"

namespace ait
{

template <template <typename, typename> class TWeakLearner, typename TSampleIterator, typename TRandomEngine = std::mt19937_64>
class LevelForestTrainer
{
public:
    struct LevelTrainingParameters : public TrainingParameters
    {
        // TODO
    };

    using ParametersT = LevelTrainingParameters;

    using SampleIteratorT = TSampleIterator;
    using SampleT = typename TSampleIterator::value_type;
    using SamplePointerT = const SampleT *;
    using SamplePointerIteratorT = typename std::vector<SamplePointerT>::const_iterator;
    using SamplePointerIteratorWrapperT = ait::PointerIteratorWrapper<SamplePointerIteratorT, const SampleT>;
//    using SamplePointerIteratorWrapperT = ait::PointerIteratorWrapper<SamplePointerIteratorT, boost::use_default>;
    
    using WeakLearnerT = TWeakLearner<SamplePointerIteratorWrapperT, TRandomEngine>;

    using StatisticsT = typename WeakLearnerT::StatisticsT;
    using SplitPointT = typename WeakLearnerT::SplitPointT;
    using ForestT = Forest<SplitPointT, StatisticsT>;
    using TreeT = Tree<SplitPointT, StatisticsT>;
    using NodeType = typename TreeT::NodeT;
    using NodeIterator = typename TreeT::NodeIterator;

protected:
    const WeakLearnerT weak_learner_;
    const LevelTrainingParameters training_parameters_;
    
    void output_spaces(std::ostream &stream, int num_of_spaces) const
    {
        for (int i = 0; i < num_of_spaces; i++)
            stream << " ";
    }

    virtual std::map<typename TreeT::NodeIterator, std::vector<SplitPointT> > sample_split_points_batch(const TreeT &tree, const std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map, TRandomEngine &rnd_engine) const
    {
        std::map<typename TreeT::NodeIterator, std::vector<SplitPointT> > split_points_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            typename WeakLearnerT::SampleIteratorT sample_it_begin = make_pointer_iterator_wrapper(map_it->second.cbegin());
            typename WeakLearnerT::SampleIteratorT sample_it_end = make_pointer_iterator_wrapper(map_it->second.cend());
            split_points_batch[map_it->first] = weak_learner_.sample_split_points(sample_it_begin, sample_it_end, rnd_engine);
        }
        return split_points_batch;
    }
    
    virtual std::map<typename TreeT::NodeIterator, SplitStatistics<StatisticsT> > compute_split_statistics_batch(const TreeT &tree, const std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map, const std::map<typename TreeT::NodeIterator, std::vector<SplitPointT> > &split_points_batch) const
    {
        std::map<typename TreeT::NodeIterator, SplitStatistics<StatisticsT> > split_statistics_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            const std::vector<SplitPointT> &split_points = split_points_batch.at(map_it->first);
            typename WeakLearnerT::SampleIteratorT sample_it_begin = make_pointer_iterator_wrapper(map_it->second.cbegin());
            typename WeakLearnerT::SampleIteratorT sample_it_end = make_pointer_iterator_wrapper(map_it->second.cend());
            split_statistics_batch[map_it->first] = weak_learner_.compute_split_statistics(sample_it_begin, sample_it_end, split_points);
        }
        return split_statistics_batch;
    }
    
    std::map<typename TreeT::NodeIterator, SplitPointT> find_best_split_point_batch(const TreeT &tree, const std::map<typename TreeT::NodeIterator, std::vector<SplitPointT> > &split_points_batch, const std::map<typename TreeT::NodeIterator, StatisticsT> &current_statistics, const std::map<typename TreeT::NodeIterator, SplitStatistics<StatisticsT> > &split_statistics_batch) const
    {
        std::map<typename TreeT::NodeIterator, SplitPointT> best_split_point_batch;
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

    std::map<typename TreeT::NodeIterator, StatisticsT> compute_statistics_batch(const TreeT &tree, std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map) const
    {
        std::map<typename TreeT::NodeIterator, StatisticsT> statistics_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            StatisticsT statistics = weak_learner_.create_statistics();
            typename WeakLearnerT::SampleIteratorT sample_it_begin = make_pointer_iterator_wrapper(map_it->second.cbegin());
            typename WeakLearnerT::SampleIteratorT sample_it_end = make_pointer_iterator_wrapper(map_it->second.cend());
            statistics.accumulate(sample_it_begin, sample_it_end);
            statistics_batch.insert(std::make_pair(map_it->first, std::move(statistics)));
        }
        // Receive statistics from rank > 0
        //dist_statistics_batch = receive();
        std::map<typename TreeT::NodeIterator, std::vector<StatisticsT> > dist_statistics_batch;
        for (auto map_it = node_to_sample_map.begin(); map_it != node_to_sample_map.end(); ++map_it)
        {
            dist_statistics_batch[map_it->first].push_back(statistics_batch.at(map_it->first));
        }
        for (auto map_it = dist_statistics_batch.begin(); map_it != dist_statistics_batch.end(); ++map_it)
        {
            StatisticsT statistics = weak_learner_.create_statistics();
            typename std::vector<StatisticsT>::const_iterator it_start = map_it->second.cbegin();
            typename std::vector<StatisticsT>::const_iterator it_end = map_it->second.cend();
            statistics.accumulate_histograms(it_start, it_end);
            map_it->first->set_statistics(statistics);
            statistics_batch.at(map_it->first) = std::move(statistics);
        }
        return statistics_batch;
    }

    std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT> > get_sample_node_map(TreeT &tree, typename TreeT::TreeLevel &tl, TSampleIterator samples_start, TSampleIterator samples_end) const
    {
        std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT> > node_to_sample_map;
        for (auto it = samples_start; it != samples_end; ++it)
        {
            // TODO: Use template meta-programming to allow tree.evaluate to be called for both samples and sample-iterators.
            typename TreeT::NodeIterator node_it = tree.evaluate(*it);
            SamplePointerT sample_ptr = &(*it);
            node_to_sample_map[node_it].push_back(sample_ptr);
        }
        for (auto node_it = tl.begin(); node_it != tl.end(); ++node_it)
        {
            // Make sure every node of the tree-level has an entry in the map
            node_to_sample_map[node_it];
        }
        return node_to_sample_map;
    }

public:
    LevelForestTrainer(const WeakLearnerT &weak_learner, const LevelTrainingParameters &training_parameters)
    : weak_learner_(weak_learner), training_parameters_(training_parameters)
    {}

    TreeT train_tree(TSampleIterator samples_start, TSampleIterator samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_tree(samples_start, samples_end, rnd_engine);
    }

    virtual void train_tree_level(TreeT &tree, size_type current_level, TSampleIterator samples_start, TSampleIterator samples_end, TRandomEngine &rnd_engine) const
    {
        typename TreeT::TreeLevel tl(tree, current_level);
        std::map<typename TreeT::NodeIterator, std::vector<SamplePointerT> > node_to_sample_map = get_sample_node_map(tree, tl, samples_start, samples_end);
        std::cout << "current_level: " << current_level << ", # nodes: " << node_to_sample_map.size() << std::endl;
        const std::map<typename TreeT::NodeIterator, StatisticsT> &current_statistics = compute_statistics_batch(tree, node_to_sample_map);
        if (current_level < training_parameters_.tree_depth)
        {
            std::map<typename TreeT::NodeIterator, std::vector<SplitPointT> > split_points_batch = sample_split_points_batch(tree, node_to_sample_map, rnd_engine);
            std::map<typename TreeT::NodeIterator, SplitStatistics<StatisticsT> > split_statistics_batch = compute_split_statistics_batch(tree, node_to_sample_map, split_points_batch);
            // Receive split statistics from rank > 0
            std::map<typename TreeT::NodeIterator, SplitPointT> best_split_point_batch = find_best_split_point_batch(tree, split_points_batch, current_statistics, split_statistics_batch);
            for (auto map_it = best_split_point_batch.begin(); map_it != best_split_point_batch.end(); ++map_it)
            {
                typename TreeT::NodeIterator node_it = map_it->first;
                node_it->set_split_point(map_it->second);
                node_it.set_leaf(false);
                node_it.left_child().set_leaf(true);
                node_it.right_child().set_leaf(true);
            }
        }
    }

    TreeT train_tree(TSampleIterator samples_start, TSampleIterator samples_end, TRandomEngine &rnd_engine) const
    {
        TreeT tree(training_parameters_.tree_depth);
        tree.get_root_iterator().set_leaf();
        std::cout << "Training tree, # samples " << (samples_end - samples_start) << std::endl;
        for (size_type current_level = 1; current_level <= training_parameters_.tree_depth; current_level++)
        {
            train_tree_level(tree, current_level, samples_start, samples_end, rnd_engine);
        }
        return tree;
    }

    ForestT train_forest(TSampleIterator samples_start, TSampleIterator samples_end, TRandomEngine &rnd_engine) const
    {
        ForestT forest;
        for (int i=0; i < training_parameters_.num_of_trees; i++)
        {
            TreeT tree = train_tree(samples_start, samples_end, rnd_engine);
            forest.add_tree(std::move(tree));
        }
        return forest;
    }

    ForestT train_forest(TSampleIterator samples_start, TSampleIterator samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_forest(samples_start, samples_end, rnd_engine);
    }

};
    
}

