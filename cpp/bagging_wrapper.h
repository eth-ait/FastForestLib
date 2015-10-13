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

    template <typename TRandomEngine, typename SampleT>
    class BaggingSampleProvider
    {
    public:
        virtual std::vector<SampleT> get_sample_bag(TRandomEngine& rnd_engine) const = 0;
    };

    // TODO: Allow bagging of meta-samples (i.e. images)
    template <template <typename> class TForestTrainer, typename SampleT>
    class BaggingWrapper
    {
    public:
        using SampleIteratorT = typename std::vector<SampleT>::const_iterator;
        using ForestTrainerT = TForestTrainer<SampleIteratorT>;
        using ForestT = typename ForestTrainerT::ForestT;
        using TreeT = typename ForestTrainerT::TreeT;
        using RandomEngineT = typename ForestTrainerT::RandomEngineT;

        BaggingWrapper(const ForestTrainerT& trainer, const BaggingSampleProvider<RandomEngineT, SampleT>& provider)
        : trainer_(trainer), provider_(provider)
        {}

        TreeT train_tree(RandomEngineT& rnd_engine) const
        {
            std::vector<SampleT> samples = provider_.get_sample_bag(rnd_engine);
            SampleIteratorT samples_start = samples.cbegin();
            SampleIteratorT samples_end = samples.cend();
            return  trainer_.train_tree(samples_start, samples_end);
        }

        TreeT train_tree() const
        {
            RandomEngineT rnd_engine;
            return train_tree(rnd_engine);
        }

        ForestT train_forest(RandomEngineT &rnd_engine) const
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
        const BaggingSampleProvider<RandomEngineT, SampleT>& provider_;
    };

}
