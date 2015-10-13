//
//  weak_learner.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

#include <tuple>
#include <iostream>
#include <random>

#include "ait.h"

namespace ait
{

template <typename TStatistics>
class SplitStatistics
{
    std::vector<TStatistics> left_statistics_collection_;
    std::vector<TStatistics> right_statistics_collection_;

public:
    SplitStatistics() {}

    template <typename TStatisticsFactory>
    SplitStatistics(size_t num_of_split_points, const TStatisticsFactory& statistics_factory)
    {
        left_statistics_collection_.resize(num_of_split_points, statistics_factory.create());
        right_statistics_collection_.resize(num_of_split_points, statistics_factory.create());
    }
    size_type size() const
    {
        return left_statistics_collection_.size();
    }

    TStatistics& get_left_statistics(size_type index)
    {
        return left_statistics_collection_[index];
    }
    
    TStatistics& get_right_statistics(size_type index)
    {
        return right_statistics_collection_[index];
    }
    
    const TStatistics& get_left_statistics(size_type index) const
    {
        return left_statistics_collection_[index];
    }

    const TStatistics& get_right_statistics(size_type index) const
    {
        return right_statistics_collection_[index];
    }

    void accumulate(const SplitStatistics& split_statistics)
    {
        assert(size() == split_statistics.size());
        for (size_type i = 0; i < size(); i++)
        {
            get_left_statistics(i).accumulate(split_statistics.get_left_statistics(i));
            get_right_statistics(i).accumulate(split_statistics.get_right_statistics(i));
        }
    }

private:
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive & left_statistics_collection_;
        archive & right_statistics_collection_;
    }
#endif
    
    friend class cereal::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("left_statistics_collection", left_statistics_collection_));
        archive(cereal::make_nvp("right_statistics_collection", right_statistics_collection_));
    }
};

template <typename TSplitPointCandidates, typename TStatisticsFactory, typename TSampleIterator, typename TRandomEngine = std::mt19937_64>
class WeakLearner
{
public:
    using SplitPointT = TSplitPointCandidates;
    using SplitPointCandidatesT = std::vector<SplitPointT>;
//    using SplitPointCandidatesT = TSplitPointCandidates;
//    using SplitPointT = typename SplitPointCandidatesT::SplitPointT;
    using StatisticsFactoryT = TStatisticsFactory;
    using StatisticsT = typename TStatisticsFactory::value_type;
    using SampleIteratorT = TSampleIterator;
    using RandomEngineT = TRandomEngine;

protected:
    StatisticsFactoryT statistics_factory_;

    virtual scalar_type compute_information_gain(const StatisticsT& current_statistics, const StatisticsT& left_statistics, const StatisticsT& right_statistics) const
    {
        scalar_type current_entropy = current_statistics.entropy();
        scalar_type left_entropy = left_statistics.entropy();
        scalar_type right_entropy = right_statistics.entropy();
        // We use a static cast here because current_statistics.NumOfSamples() usually returns an integer.
        scalar_type information_gain = current_entropy
        - (left_statistics.num_of_samples() * left_entropy + right_statistics.num_of_samples() * right_entropy) / static_cast<scalar_type>(current_statistics.num_of_samples());
        return information_gain;
    }

public:
    WeakLearner(const StatisticsFactoryT& statistics_factory)
    : statistics_factory_(statistics_factory)
    {}

    virtual ~WeakLearner()
    {}

    StatisticsT create_statistics() const
    {
        return statistics_factory_.create();
    }

    // Has to be implemented by a base class
    virtual SplitPointCandidatesT sample_split_points(SampleIteratorT first_sample, SampleIteratorT last_sample, RandomEngineT& rnd_engine) const = 0;
    
    // Has to be implemented by a base class
    virtual SplitStatistics<StatisticsT> compute_split_statistics(SampleIteratorT first_sample, SampleIteratorT last_sample, const SplitPointCandidatesT& split_points) const = 0;

    StatisticsT compute_statistics(SampleIteratorT first_sample, SampleIteratorT last_sample) const
    {
        StatisticsT statistics = statistics_factory_.create();
        for (SampleIteratorT sample_it=first_sample; sample_it != last_sample; sample_it++) {
            statistics.lazy_accumulate(*sample_it);
        }
        statistics.finish_lazy_accumulation();
        return statistics;
    }

    virtual std::tuple<size_type, scalar_type> find_best_split_point_tuple(const StatisticsT& current_statistics, const SplitStatistics<StatisticsT>& split_statistics) const {
        size_type best_split_point_index = 0;
        scalar_type best_information_gain = -std::numeric_limits<scalar_type>::infinity();
        for (size_type i = 0; i < split_statistics.size(); i++)
        {
            scalar_type information_gain = compute_information_gain(current_statistics,
                                                                   split_statistics.get_left_statistics(i),
                                                                   split_statistics.get_right_statistics(i));
            if (information_gain > best_information_gain) {
                best_information_gain = information_gain;
                best_split_point_index = i;
            }
        }
        return std::make_tuple(best_split_point_index, best_information_gain);
    }

    SampleIteratorT partition(SampleIteratorT first_sample, SampleIteratorT last_sample, const SplitPointT& split_point) const {
        SampleIteratorT it_left = first_sample;
        SampleIteratorT it_right = last_sample - 1;
        while (it_left < it_right) {
            Direction direction = split_point.evaluate(*it_left);
            if (direction == Direction::LEFT)
                it_left++;
            else {
                std::swap(*it_left, *it_right);
                it_right--;
            }
        }

        Direction direction = split_point.evaluate(*it_left);
        SampleIteratorT it_split;
        if (direction == Direction::LEFT)
            it_split = it_left + 1;
        else
            it_split = it_left;

        // Verify partitioning
        for (SampleIteratorT it = first_sample; it != it_split; it++)
        {
            Direction dir = split_point.evaluate(*it);
            if (dir != Direction::LEFT)
                throw std::runtime_error("Samples are not partitioned properly.");
        }
        for (SampleIteratorT it = it_split; it != last_sample; it++)
        {
            Direction dir = split_point.evaluate(*it);
            if (dir != Direction::RIGHT)
                throw std::runtime_error("Samples are not partitioned properly.");
        }

        return it_split;
    }

};

}
