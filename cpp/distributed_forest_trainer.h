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
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/utility/enable_if.hpp>

#include "ait.h"
#include "forest_trainer.h"

namespace ait
{

struct DistributedTrainingParameters : public TrainingParameters
{
    // TODO
};

// TODO
template <typename BaseIterator>
    class PointerIteratorWrapper : public boost::iterator_adaptor<PointerIteratorWrapper<BaseIterator>, BaseIterator, typename BaseIterator::value_type::element_type>
{
protected:
    BaseIterator it_;

public:
    PointerIteratorWrapper()
    : PointerIteratorWrapper::iterator_adapter_(0)
    {}

    explicit PointerIteratorWrapper(BaseIterator it)
    : PointerIteratorWrapper::iterator_adaptor_(it)
    {}

    template <typename OtherBaseIterator>
    PointerIteratorWrapper(
        const PointerIteratorWrapper<OtherBaseIterator> &other,
        typename boost::enable_if<
            boost::is_convertible<OtherBaseIterator, BaseIterator>, int>::type = 0
    )
    : PointerIteratorWrapper::iterator_adaptor_(other.base())
    {}

private:
    friend class boost::iterator_core_access;
    typename PointerIteratorWrapper::iterator_adaptor_::reference dereference() const
    {
        return *(*this->base());
    }
};

template <typename BaseIterator>
inline PointerIteratorWrapper<BaseIterator> make_pointer_iterator_wrapper(BaseIterator it)
{
    return PointerIteratorWrapper<BaseIterator>(it);
}

template <typename TSamplePointerIterator, typename TWeakLearner, typename TRandomEngine = std::mt19937_64>
class DistributedForestTrainer
{
public:
    using SamplePointerT = typename TSamplePointerIterator::value_type;
    using StatisticsT = typename TWeakLearner::StatisticsT;
    using SampleIteratorT = typename TWeakLearner::SampleIteratorT;
    using SplitPointT = typename TWeakLearner::SplitPointT;
    using TreeType = Tree<SplitPointT, StatisticsT>;
    using NodeType = typename TreeType::NodeType;
    using NodeIterator = typename TreeType::NodeIterator;

protected:
    const TWeakLearner weak_learner_;
    const TrainingParameters training_parameters_;
    
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

    std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > sample_split_points_batch(const std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > &node_to_sample_map, TRandomEngine &rnd_engine) const
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
            statistics_batch[node_it] = StatisticsT::accumulate(sample_it_begin, sample_it_end);;
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
            StatisticsT statistics = StatisticsT::accumulate_histograms(it_start, it_end);
            typename TreeType::NodeIterator node_it = map_it->first;
            node_it->set_statistics(statistics);
            statistics_batch.at(map_it->first) = statistics;
        }
        return statistics_batch;
    }

    std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > get_sample_node_map(TreeType &tree, TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, size_type current_depth) const
    {
        std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > node_to_sample_map;
        for (auto it = samples_start; it != samples_end; ++it)
        {
            // TODO: Use template meta-programming to allow tree.evaluate to be called for both samples and sample-iterators.
            typename TreeType::NodeIterator node_it = tree.evaluate(*(*it), current_depth);
            node_to_sample_map[node_it].push_back(*it);
        }
        return node_to_sample_map;
    }
    
public:
    DistributedForestTrainer(const TWeakLearner &weak_learner, const TrainingParameters &training_parameters)
    : weak_learner_(weak_learner), training_parameters_(training_parameters)
    {}

    Tree<SplitPointT, StatisticsT> train_tree(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_tree(samples_start, samples_end, rnd_engine);
    }

    Tree<SplitPointT, StatisticsT> train_tree(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, TRandomEngine &rnd_engine) const
    {
        TreeType tree(training_parameters_.tree_depth);
        std::cout << "Training tree, # samples " << (samples_end - samples_start) << std::endl;
        for (size_type current_depth = 1; current_depth <= training_parameters_.tree_depth; current_depth++)
        {
            typename TreeType::TreeLevel tl(tree, current_depth);
            std::cout << "current_depth: " << current_depth << ", # nodes: " << tl.size() << std::endl;
            std::map<typename TreeType::NodeIterator, std::vector<SamplePointerT> > node_to_sample_map = get_sample_node_map(tree, samples_start, samples_end, current_depth);
            const std::map<typename TreeType::NodeIterator, StatisticsT> &current_statistics = compute_statistics_batch(node_to_sample_map);
            if (current_depth < training_parameters_.tree_depth)
            {
                std::map<typename TreeType::NodeIterator, std::vector<SplitPointT> > split_points_batch = sample_split_points_batch(node_to_sample_map, rnd_engine);
                // Distribute split points
                std::map<typename TreeType::NodeIterator, SplitStatistics<StatisticsT> > split_statistics_batch = compute_split_statistics_batch(node_to_sample_map, split_points_batch);
                // Receive split statistics from rank > 0
                std::map<typename TreeType::NodeIterator, SplitPointT> best_split_point_batch = find_best_split_point_batch(split_points_batch, current_statistics, split_statistics_batch);
                for (auto map_it = best_split_point_batch.begin(); map_it != best_split_point_batch.end(); ++map_it)
                {
                    typename TreeType::NodeIterator node_it = map_it->first;
                    node_it->set_split_point(map_it->second);
                }
            }
            // save new tree
            // distribute new tree
        }
        return tree;
    }
    
    Forest<SplitPointT, StatisticsT> train_forest(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end) const
    {
        TRandomEngine rnd_engine;
        return train_forest(samples_start, samples_end, rnd_engine);
    }
    
    Forest<SplitPointT, StatisticsT> train_forest(TSamplePointerIterator samples_start, TSamplePointerIterator samples_end, TRandomEngine &rnd_engine) const
    {
        Forest<SplitPointT, StatisticsT> forest;
        for (int i=0; i < training_parameters_.num_of_trees; i++)
        {
            Tree<SplitPointT, StatisticsT> tree = train_tree(samples_start, samples_end, rnd_engine);
            forest.add_tree(std::move(tree));
        }
        return forest;
    }
};
    
}
