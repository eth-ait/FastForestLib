#pragma once
#include <iostream>
#include <sstream>

#include "ait.h"
#include "forest.h"
#include "weak_learner.h"

namespace ait
{

class TrainingParameters
{
public:
    int num_of_trees() const
    {
        return 1;
    }
    int tree_depth() const
    {
        return 10;
    }
    int minimum_num_of_samples() const
    {
        return 100;
    }
    double minimum_information_gain() const
    {
        return 0.0;
    }
};

template <typename TSample, typename TWeakLearner, typename TTrainingParameters, typename TRandomEngine = std::mt19937_64>
class ForestTrainer
{
public:
    typedef typename TWeakLearner::StatisticsT StatisticsT;
    typedef typename TWeakLearner::SampleIteratorT SampleIteratorT;
    typedef typename TWeakLearner::SplitPointT SplitPointT;
    typedef typename Tree<SplitPointT, StatisticsT>::NodeIterator NodeIterator;

private:
    const TWeakLearner weak_learner_;
    const TTrainingParameters training_parameters_;

    void output_spaces(std::ostream &stream, int num_of_spaces) const
    {
        for (int i = 0; i < num_of_spaces; i++)
            stream << " ";
    }

public:
    ForestTrainer(const TWeakLearner &weak_learner, const TTrainingParameters &training_parameters)
        : weak_learner_(weak_learner), training_parameters_(training_parameters)
    {}

    void train_tree_recursive(NodeIterator node_iter, SampleIteratorT i_start, SampleIteratorT i_end, TRandomEngine &rnd_engine, int current_depth = 1) const
    {
        // TODO: Remove io operations
//			std::ostringstream o_stream;
//			for (int i = 0; i < current_depth; i++)
//				o_stream << " ";
//			const std::string prefix = o_stream.str();
        output_spaces(std::cout, current_depth - 1);
        std::cout << "depth: " << current_depth << ", samples: " << (i_end - i_start) << std::endl;

        // assign statistics to node
        typename TWeakLearner::StatisticsT statistics = weak_learner_.compute_statistics(i_start, i_end);
        node_iter->set_statistics(statistics);

        // stop splitting the node if the minimum number of samples has been reached
        if (i_end - i_start < training_parameters_.minimum_num_of_samples()) {
            //node.leaf_node = True
            output_spaces(std::cout, current_depth - 1);
            std::cout << "Minimum number of samples. Stopping." << std::endl;
            return;
        }

        // stop splitting the node if it is a leaf node
        if (node_iter.is_leaf_node()) {
            output_spaces(std::cout, current_depth - 1);
            std::cout << "Reached leaf node. Stopping." << std::endl;
            return;
        }

        std::vector<SplitPointT> split_points = weak_learner_.sample_split_points(i_start, i_end, rnd_engine);

        // TODO: distribute features and thresholds to ranks > 0

        // compute the statistics for all feature and threshold combinations
        SplitStatistics<typename TWeakLearner::StatisticsT> split_statistics = weak_learner_.compute_split_statistics(i_start, i_end, split_points);

        // TODO: send statistics to rank 0
        // send split_statistics.get_buffer()

        // TODO: receive statistics from rank > 0
        // for received statistics
            // split_statistics.accumulate(statistics)

        // find the best feature(only on rank 0)
        std::tuple<size_type, scalar_type> best_split_point_tuple = weak_learner_.find_best_split_point(statistics, split_statistics);

        // TODO: send best feature, threshold and information gain to ranks > 0

        scalar_type best_information_gain = std::get<1>(best_split_point_tuple);
        // TODO: move criterion into trainingContext ?
        // stop splitting the node if the best information gain is below the minimum information gain
        if (best_information_gain < training_parameters_.minimum_information_gain()) {
            //node.leaf_node = True
            output_spaces(std::cout, current_depth - 1);
            std::cout << "Too little information gain. Stopping." << std::endl;
            return;
        }

        // partition sample_indices according to the selected feature and threshold.
        // i.e.sample_indices[:i_split] will contain the left child indices
        // and sample_indices[i_split:] will contain the right child indices
        size_type best_split_point_index = std::get<0>(best_split_point_tuple);
        SplitPointT best_split_point = split_points[best_split_point_index];
        SampleIteratorT i_split = weak_learner_.partition(i_start, i_end, best_split_point);

        node_iter->set_split_point(best_split_point);

        // TODO: can we reuse computed statistics from split_point_context ? ? ?
        //left_child_statistics = None
        //right_child_statistics = None

        // train left and right child
        //print("{}Going left".format(prefix))
        train_tree_recursive(node_iter.left_child(), i_start, i_split, rnd_engine, current_depth + 1);
        //print("{}Going right".format(prefix))
        train_tree_recursive(node_iter.right_child(), i_split, i_end, rnd_engine, current_depth + 1);
    }

    Tree<SplitPointT, StatisticsT> train_tree(std::vector<TSample> &samples) const
    {
        TRandomEngine rnd_engine;
        return train_tree(samples, rnd_engine);
    }
    

    Tree<SplitPointT, StatisticsT> train_tree(std::vector<TSample> &samples, TRandomEngine &rnd_engine) const
    {
        Tree<SplitPointT, StatisticsT> tree(training_parameters_.tree_depth());
        train_tree_recursive(tree.get_root(), samples.begin(), samples.end(), rnd_engine);
        return tree;
    }
    
    Forest<SplitPointT, StatisticsT> train_forest(std::vector<TSample> &samples) const
    {
        TRandomEngine rnd_engine;
        return train_forest(samples, rnd_engine);
    }

    Forest<SplitPointT, StatisticsT> train_forest(std::vector<TSample> &samples, TRandomEngine &rnd_engine) const
    {
        Forest<SplitPointT, StatisticsT> forest;
        for (int i=0; i < training_parameters_.num_of_trees(); i++) {
            Tree<SplitPointT, StatisticsT> tree = train_tree(samples, rnd_engine);
            forest.add_tree(std::move(tree));
        }
        return forest;
    }

};

template <typename TSample, typename TWeakLearner, typename TTrainingParameters, typename TRandomEngine = std::mt19937_64>
class DistributedForestTrainer
{
public:
    typedef typename TWeakLearner::StatisticsT StatisticsT;
    typedef typename TWeakLearner::SampleIteratorT SampleIteratorT;
    typedef typename TWeakLearner::SplitPointT SplitPointT;
    typedef typename Tree<SplitPointT, StatisticsT>::NodeIterator NodeIterator;

private:
    const TWeakLearner weak_learner_;
    const TTrainingParameters training_parameters_;
    
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
            std::vector<SplitPointT> split_points = weak_learner_.sample_split_points(i_start, i_end, rnd_engine);
            split_points_batch.insert(split_points_batch.end(),
                                      std::make_move_iterator(split_points.begin()),
                                      std::make_move_iterator(split_points.end())
            );
        }
        return split_points_batch;
    }

public:
    DistributedForestTrainer(const TWeakLearner &weak_learner, const TTrainingParameters &training_parameters)
    : weak_learner_(weak_learner), training_parameters_(training_parameters)
    {}

    void train_nodes(const std::vector<NodeIterator> &node_iters, SampleIteratorT i_start, SampleIteratorT i_end, TRandomEngine &rnd_engine) const
    {
        std::cout << "nodes: " << node_iters.size() << ", samples: " << (i_end - i_start) << std::endl;

        // assign statistics to nodes
        for (auto it = node_iters.begin(); it != node_iters.end(); it++)
        {
            typename TWeakLearner::StatisticsT statistics = weak_learner_.compute_statistics(i_start, i_end);
            (*it)->set_statistics(statistics);
        }
        
        // stop splitting the node if the minimum number of samples has been reached
        if (i_end - i_start < training_parameters_.minimum_num_of_samples()) {
            //node.leaf_node = True
            std::cout << "Minimum number of samples. Stopping." << std::endl;
            return;
        }

        // stop splitting the node if it is a leaf node
        for (auto it = node_iters.begin(); it != node_iters.end(); it++)
        {
            assert(!it->is_leaf_node());
        }

        std::vector<SplitPointT> split_points = sample_split_points_batch(node_iters.size(), i_start, i_end, rnd_engine);
        
        // TODO: distribute features and thresholds to ranks > 0
        
        // compute the statistics for all feature and threshold combinations
        SplitStatistics<typename TWeakLearner::StatisticsT> split_statistics = weak_learner_.compute_split_statistics(i_start, i_end, split_points);

        // TODO: send statistics to rank 0
        // send split_statistics.get_buffer()

        // TODO: receive statistics from rank > 0
        // for received statistics
        // split_statistics.accumulate(statistics)
        
        // find the best feature(only on rank 0)
        std::tuple<size_type, scalar_type> best_split_point_tuple = weak_learner_.find_best_split_point(statistics, split_statistics);
        
        // TODO: send best feature, threshold and information gain to ranks > 0
        
        scalar_type best_information_gain = std::get<1>(best_split_point_tuple);
        // TODO: move criterion into trainingContext ?
        // stop splitting the node if the best information gain is below the minimum information gain
        if (best_information_gain < training_parameters_.minimum_information_gain()) {
            //node.leaf_node = True
            output_spaces(std::cout, current_depth - 1);
            std::cout << "Too little information gain. Stopping." << std::endl;
            return;
        }

        // partition sample_indices according to the selected feature and threshold.
        // i.e.sample_indices[:i_split] will contain the left child indices
        // and sample_indices[i_split:] will contain the right child indices
        size_type best_split_point_index = std::get<0>(best_split_point_tuple);
        SplitPointT best_split_point = split_points[best_split_point_index];
        SampleIteratorT i_split = weak_learner_.partition(i_start, i_end, best_split_point);
        
        node_iter->set_split_point(best_split_point);

        // train left and right child
        train_tree_recursive(node_iter.left_child(), i_start, i_split, rnd_engine, current_depth + 1);
        train_tree_recursive(node_iter.right_child(), i_split, i_end, rnd_engine, current_depth + 1);
    }

};

}
