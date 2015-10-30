//
//  distributed_bagging_wrapper.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 30/10/15.
//
//

#pragma once

//
//  bagging_wrapper.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 13/10/15.
//
//

#pragma once

#include <algorithm>

#include <boost/mpi/communicator.hpp>
#ifdef SERIALIZE_WITH_BOOST
#include <boost/serialization/vector.hpp>
#include "serialization_utils.h"
#endif

namespace ait
{

/// @brief A wrapper for a distributed forest trainer that allows bagging of the samples before training.
template <template <typename> class TForestTrainer, typename TSampleProvider>
class DistributedBaggingWrapper
{
public:
    using SampleProviderT = TSampleProvider;
    using SampleIteratorT = typename SampleProviderT::SampleIteratorT;
    using ForestTrainerT = TForestTrainer<SampleIteratorT>;
    using ForestT = typename ForestTrainerT::ForestT;
    using TreeT = typename ForestTrainerT::TreeT;
    using RandomEngineT = typename ForestTrainerT::RandomEngineT;

    DistributedBaggingWrapper(const boost::mpi::communicator& comm, const ForestTrainerT& trainer, SampleProviderT& provider)
    : comm_(comm), trainer_(trainer), provider_(provider)
    {}

    TreeT train_tree(RandomEngineT& rnd_engine) const
    {
        // TODO: Introduce memory management of images.
        size_type num_of_samples = provider_.get_num_of_samples();
        std::vector<size_type> bag_indices;
        if (comm_.rank() == 0)
        {
            bag_indices = provider_.get_sample_bag_indices(rnd_engine);
            std::sort(bag_indices.begin(), bag_indices.end());
        }
        broadcast_vector(bag_indices);
        // Split samples indices among all computing nodes.
        size_type index_start = compute_split_start_index(comm_.rank(), num_of_samples);
        size_type index_end = compute_split_start_index(comm_.rank() + 1, num_of_samples) - 1;
        ait::log_debug() << "index_start: " << index_start << ", index_end: " << index_end;
        auto bag_it_start = std::lower_bound(bag_indices.cbegin(), bag_indices.cend(), index_start);
        auto bag_it_stop = std::upper_bound(bag_indices.cbegin(), bag_indices.cend(), index_end);
        ait::log_debug() << "bag_it_start: " << (bag_it_start - bag_indices.cbegin()) << ", bag_it_end: " << (bag_it_stop - bag_indices.cbegin());
        std::vector<size_type> selected_indices(bag_it_start, bag_it_stop);
        // Load samples.
        provider_.load_samples(selected_indices);
        SampleIteratorT samples_start = provider_.get_samples_begin();
        SampleIteratorT samples_end = provider_.get_samples_end();
        // Train tree.
        return  trainer_.train_tree(samples_start, samples_end);
    }

    TreeT train_tree() const
    {
        RandomEngineT rnd_engine;
        return train_tree(rnd_engine);
    }

    ForestT train_forest(RandomEngineT& rnd_engine) const
    {
        ForestT forest;
        for (int i=0; i < trainer_.get_parameters().num_of_trees; i++)
        {
            TreeT tree = train_tree(rnd_engine);
            forest.add_tree(std::move(tree));
        }
        return forest;
    }

    ForestT train_forest() const
    {
        RandomEngineT rnd_engine;
        return train_forest(rnd_engine);
    }

private:
    template <typename T>
    void broadcast_vector(std::vector<T>& vec, int root = 0) const
    {
        broadcast(comm_, vec, root);
    }

    size_type compute_split_start_index(int rank, size_type num_of_samples) const
    {
        return static_cast<size_type>(rank * num_of_samples / static_cast<double>(comm_.size()));
    }

    boost::mpi::communicator comm_;
    const ForestTrainerT& trainer_;
    SampleProviderT& provider_;
};

}
