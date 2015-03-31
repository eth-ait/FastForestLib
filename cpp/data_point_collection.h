#ifndef AITDistributedRandomForest_data_point_collection_h
#define AITDistributedRandomForest_data_point_collection_h

#include <vector>


template<typename size_type=std::size_t>
class DataPointCollection {
public:

    virtual size_type Count() const = 0;

};

#endif
