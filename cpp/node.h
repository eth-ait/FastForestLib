//
//  node.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

#include <memory>
#include <vector>
#include <cereal/cereal.hpp>

namespace ait
{
    
enum class Direction { LEFT = -1, RIGHT = +1 };

/// @brief A node of a decision tree.
template <typename TSplitPoint, typename TStatistics>
class Node
{
public:
    Node()
    {}

    ~Node()
    {}

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
    
private:
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;
#else
    friend class cereal::access;
#endif

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version)
    {
#ifdef SERIALIZE_WITH_BOOST
        archive & split_point_;
        archive & statistics_;
#else
        archive(cereal::make_nvp("split_point", split_point_));
        archive(cereal::make_nvp("statistics", statistics_));
#endif
    }
    
    TSplitPoint split_point_;
    TStatistics statistics_;
};

}
