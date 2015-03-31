#ifndef AITDistributedRandomForest_split_point_h
#define AITDistributedRandomForest_split_point_h

#include <vector>

#include "data_point_collection.h"


namespace AIT {

	enum class Direction { LEFT = -1, RIGHT = +1 };

	//template<typename size_type = std::size_t>
	//class SplitPoint {
	//public:
	//	virtual Direction Evaluate(const DataPointCollection &data_point_collection, size_type index) = 0;

	//};

}

#endif
