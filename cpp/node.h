#pragma once

#include <memory>
#include <vector>


namespace ait
{
    
enum class Direction { LEFT = -1, RIGHT = +1 };

/// @brief A node of a decision tree.
template <typename TSplitPoint, typename TStatistics>
class Node
{
    TSplitPoint split_point_;
    TStatistics statistics_;

public:
    typedef std::vector<int>::size_type size_type;

    ~Node() {}

    const TSplitPoint & get_split_point() const
    {
        return split_point_;
    }

    void set_split_point(const TSplitPoint &split_point)
    {
        split_point_ = split_point;
    }

    const TStatistics & get_statistics() const
    {
        return statistics_;
    }

    void set_statistics(const TStatistics &statistics)
    {
        statistics_ = statistics;
    }
    
    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version)
    {
        archive(cereal::make_nvp("split_point", split_point_));
        archive(cereal::make_nvp("statistics", statistics_));
    }

};

}
