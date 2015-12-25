//
//  bagging_wrapper.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 13/10/15.
//
//

#pragma once

namespace ait
{

/// @brief A wrapper for a forest trainer that allows bagging of the samples before training.
template <template <typename> class TForestTrainer, typename TSampleProvider>
class BaggingWrapper
{
public:
    using SampleProviderT = TSampleProvider;
    using SampleIteratorT = typename SampleProviderT::SampleIteratorT;
    using ForestTrainerT = TForestTrainer<SampleIteratorT>;
    using ForestT = typename ForestTrainerT::ForestT;
    using TreeT = typename ForestTrainerT::TreeT;
    using RandomEngineT = typename ForestTrainerT::RandomEngineT;

    BaggingWrapper(const ForestTrainerT& trainer, SampleProviderT& provider)
    : trainer_(trainer), provider_(provider)
    {}

    TreeT train_tree(RandomEngineT& rnd_engine) const
    {
        provider_.load_sample_bag(rnd_engine);
        SampleIteratorT samples_start = provider_.get_samples_begin();
        SampleIteratorT samples_end = provider_.get_samples_end();
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
    const ForestTrainerT& trainer_;
    SampleProviderT& provider_;
};


}
