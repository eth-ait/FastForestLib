#ifndef AITDistributedRandomForest_split_point_h
#define AITDistributedRandomForest_split_point_h

#include <vector>

#include "data_point_collection.h"


template<typename value_type=double, typename size_type=std::size_t>
class SplitPoint {
public:
    enum class Direction {LEFT, RIGHT};

    virtual Direction Evaluate(const DataPointCollection &data_point_collection, size_type index) = 0;
    
};

#endif
