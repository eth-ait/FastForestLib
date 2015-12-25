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
    using SampleBagBatchT = typename SampleProviderT::SampleBagBatchT;
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
    	std::vector<SampleBagBatchT> sample_bag_batches;
    	if (comm_.rank() == 0)
    	{
    		sample_bag_batches = provider_.compute_sample_bag_batches(comm_.size(), rnd_engine);
    	}
    	SampleBagBatchT sample_batch = scatter_vector(sample_bag_batches);
		provider_.load_sample_batch(sample_batch, rnd_engine);
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
    T scatter_vector(std::vector<T>& vec, int root = 0) const
    {
    	T out_value;
    	scatter(comm_, vec, out_value, root);
    	return out_value;
    }

    boost::mpi::communicator comm_;
    const ForestTrainerT& trainer_;
    SampleProviderT& provider_;
};

}
