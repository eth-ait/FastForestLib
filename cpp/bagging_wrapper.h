//
//  bagging_wrapper.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 13/10/15.
//
//

#pragma once

#include <memory>

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

    BaggingWrapper(const ForestTrainerT& trainer, const std::shared_ptr<SampleProviderT>& provider_ptr)
    : trainer_(trainer), provider_ptr_(provider_ptr)
    {}

    TreeT train_tree(RandomEngineT& rnd_engine) const
    {
        ait::log_info(false) << "Creating sample bag ... " << std::flush;
        provider_ptr_->load_sample_bag(rnd_engine);
        ait::log_info(false) << " Done." << std::endl;
        SampleIteratorT samples_start = provider_ptr_->get_samples_begin();
        SampleIteratorT samples_end = provider_ptr_->get_samples_end();
        ait::log_info(false) << "Num of samples: " << (samples_end - samples_start) << std::endl;
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
        for (size_type i=0; i < trainer_.get_parameters().num_of_trees; i++)
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
    const std::shared_ptr<SampleProviderT> provider_ptr_;
};


}
