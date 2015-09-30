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

    SplitStatistics(size_t num_of_split_points)
    {
        left_statistics_collection_.resize(num_of_split_points);
        right_statistics_collection_.resize(num_of_split_points);
    }
    size_type size() const
    {
        return left_statistics_collection_.size();
    }

    TStatistics & get_left_statistics(size_type index)
    {
        return left_statistics_collection_[index];
    }
    
    TStatistics & get_right_statistics(size_type index)
    {
        return right_statistics_collection_[index];
    }
    
    const TStatistics & get_left_statistics(size_type index) const
    {
        return left_statistics_collection_[index];
    }
    
    const TStatistics & get_right_statistics(size_type index) const
    {
        return right_statistics_collection_[index];
    }

};

template <typename TSplitPoint, typename TStatistics, typename TSampleIterator, typename TRandomEngine = std::mt19937_64>
class WeakLearner
{
public:
    using SplitPointT = TSplitPoint;
    using StatisticsT = TStatistics;
    using SampleIteratorT = TSampleIterator;

protected:
    virtual scalar_type compute_information_gain(const TStatistics &current_statistics, const TStatistics &left_statistics, const TStatistics &right_statistics) const {
        scalar_type current_entropy = current_statistics.entropy();
        scalar_type left_entropy = left_statistics.entropy();
        scalar_type right_entropy = right_statistics.entropy();
        // We use a static cast here because current_statistics.NumOfSamples() usually returns an integer.
        scalar_type information_gain = current_entropy
        - (left_statistics.num_of_samples() * left_entropy + right_statistics.num_of_samples() * right_entropy) / static_cast<scalar_type>(current_statistics.num_of_samples());
        return information_gain;
    }

public:
    WeakLearner()
    {}

    virtual ~WeakLearner()
    {}

    // Has to be implemented by a base class
    virtual std::vector<TSplitPoint> sample_split_points(TSampleIterator first_sample, TSampleIterator last_sample, TRandomEngine &rnd_engine) const = 0;
    
    // Has to be implemented by a base class
    virtual SplitStatistics<TStatistics> compute_split_statistics(TSampleIterator first_sample, TSampleIterator last_sample, const std::vector<TSplitPoint> &split_points) const = 0;

    TStatistics compute_statistics(TSampleIterator first_sample, TSampleIterator last_sample) const
    {
        TStatistics statistics;
        for (TSampleIterator sample_it=first_sample; sample_it != last_sample; sample_it++) {
            statistics.lazy_accumulate(*sample_it);
        }
        statistics.finish_lazy_accumulation();
        return statistics;
    }

    virtual std::tuple<size_type, scalar_type> find_best_split_point(const TStatistics &current_statistics, const SplitStatistics<TStatistics> &split_statistics) const {
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

    TSampleIterator partition(TSampleIterator first_sample, TSampleIterator last_sample, const TSplitPoint &split_point) const {
        TSampleIterator it_left = first_sample;
        TSampleIterator it_right = last_sample - 1;
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
        TSampleIterator it_split;
        if (direction == Direction::LEFT)
            it_split = it_left + 1;
        else
            it_split = it_left;

        // Verify partitioning
        for (TSampleIterator it = first_sample; it != it_split; it++)
        {
            Direction dir = split_point.evaluate(*it);
            if (dir != Direction::LEFT)
                throw std::runtime_error("Samples are not partitioned properly.");
        }
        for (TSampleIterator it = it_split; it != last_sample; it++)
        {
            Direction dir = split_point.evaluate(*it);
            if (dir != Direction::RIGHT)
                throw std::runtime_error("Samples are not partitioned properly.");
        }

        return it_split;
    }

};

// TODO
//    friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
//    virtual void writeToStream(std::ostream &os);
//    std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}
