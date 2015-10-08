#pragma once
#include <iostream>
#include <sstream>
#include <boost/foreach.hpp>

#include "ait.h"
#include "forest.h"
#include "training.h"
#include "weak_learner.h"

namespace ait
{
    
template <template <typename, typename> class TWeakLearner, typename TSampleIterator, typename TRandomEngine = std::mt19937_64>
class DepthForestTrainer
{
public:
    using ParametersT = TrainingParameters;

    using SampleIteratorT = TSampleIterator;
    using SampleT = typename TSampleIterator::value_type;
    
    using WeakLearnerT = TWeakLearner<SampleIteratorT, TRandomEngine>;
    
    using StatisticsT = typename WeakLearnerT::StatisticsT;
    using SplitPointT = typename WeakLearnerT::SplitPointT;
    using ForestT = Forest<SplitPointT, StatisticsT>;
    using TreeT = Tree<SplitPointT, StatisticsT>;
    using NodeType = typename TreeT::NodeT;
    using NodeIterator = typename TreeT::NodeIterator;

private:
    const WeakLearnerT weak_learner_;
    const ParametersT training_parameters_;

    void output_spaces(std::ostream &stream, int num_of_spaces) const
    {
        for (int i = 0; i < num_of_spaces; i++)
            stream << " ";
    }

public:
    DepthForestTrainer(const WeakLearnerT &weak_learner, const ParametersT &training_parameters)
        : weak_learner_(weak_learner), training_parameters_(training_parameters)
    {}

    void train_tree_recursive(NodeIterator tree_iter, SampleIteratorT samples_start, SampleIteratorT samples_end, TRandomEngine &rnd_engine, int current_depth = 1) const
    {
        output_spaces(std::cout, current_depth - 1);
        std::cout << "depth: " << current_depth << ", samples: " << (samples_end - samples_start) << std::endl;

        // Assign statistics to node
        StatisticsT statistics = weak_learner_.compute_statistics(samples_start, samples_end);
        tree_iter->set_statistics(statistics);

        // Stop splitting the node if the minimum number of samples has been reached
        if (samples_end - samples_start < training_parameters_.minimum_num_of_samples)
        {
            tree_iter.set_leaf();
            output_spaces(std::cout, current_depth - 1);
            std::cout << "Minimum number of samples. Stopping." << std::endl;
            return;
        }

        // Stop splitting the node if it is a leaf node
        if (tree_iter.is_leaf())
        {
            output_spaces(std::cout, current_depth - 1);
            std::cout << "Reached leaf node. Stopping." << std::endl;
            return;
        }

        std::vector<SplitPointT> split_points = weak_learner_.sample_split_points(samples_start, samples_end, rnd_engine);

        // Compute the statistics for all split points
        SplitStatistics<StatisticsT> split_statistics = weak_learner_.compute_split_statistics(samples_start, samples_end, split_points);

        // Find the best split point
        std::tuple<size_type, scalar_type> best_split_point_tuple = weak_learner_.find_best_split_point_tuple(statistics, split_statistics);

        scalar_type best_information_gain = std::get<1>(best_split_point_tuple);
        // Stop splitting the node if the best information gain is below the minimum information gain
        // TODO: Introduce leaf flag in nodes or copy statistics into each child node?
        if (best_information_gain < training_parameters_.minimum_information_gain)
        {
            tree_iter.set_leaf();
            output_spaces(std::cout, current_depth - 1);
            std::cout << "Too little information gain. Stopping." << std::endl;
            return;
        }

        // Partition sample_indices according to the selected feature and threshold.
        // i.e.sample_indices[:i_split] will contain the left child indices
        // and sample_indices[i_split:] will contain the right child indices
        size_type best_split_point_index = std::get<0>(best_split_point_tuple);
        SplitPointT best_split_point = split_points[best_split_point_index];
        SampleIteratorT i_split = weak_learner_.partition(samples_start, samples_end, best_split_point);

        tree_iter->set_split_point(best_split_point);

        // TODO: Can we reuse computed statistics from split_point_context ? ? ?
        //left_child_statistics = None
        //right_child_statistics = None

        // Train left and right child
        //print("{}Going left".format(prefix))
        train_tree_recursive(tree_iter.left_child(), samples_start, i_split, rnd_engine, current_depth + 1);
        //print("{}Going right".format(prefix))
        train_tree_recursive(tree_iter.right_child(), i_split, samples_end, rnd_engine, current_depth + 1);
    }

    Tree<SplitPointT, StatisticsT> train_tree(SampleIteratorT samples_start, SampleIteratorT samples_end, TRandomEngine &rnd_engine) const
    {
        Tree<SplitPointT, StatisticsT> tree(training_parameters_.tree_depth);
        train_tree_recursive(tree.get_root_iterator(), samples_start, samples_end, rnd_engine);
        return tree;
    }
    
    Tree<SplitPointT, StatisticsT> train_tree(SampleIteratorT samples_start, SampleIteratorT samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_tree(samples_start, samples_end, rnd_engine);
    }
    
    Forest<SplitPointT, StatisticsT> train_forest(SampleIteratorT samples_start, SampleIteratorT samples_end, TRandomEngine &rnd_engine) const
    {
        Forest<SplitPointT, StatisticsT> forest;
        for (int i=0; i < training_parameters_.num_of_trees; i++)
        {
            Tree<SplitPointT, StatisticsT> tree = train_tree(samples_start, samples_end, rnd_engine);
            forest.add_tree(std::move(tree));
        }
        return forest;
    }
    
    Forest<SplitPointT, StatisticsT> train_forest(SampleIteratorT samples_start, SampleIteratorT samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_forest(samples_start, samples_end, rnd_engine);
    }
};

}
