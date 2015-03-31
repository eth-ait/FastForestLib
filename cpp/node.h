#ifndef AITDistributedRandomForest_node_h
#define AITDistributedRandomForest_node_h

#include <memory>
#include <vector>

#include "data_point_collection.h"
#include "split_point.h"

namespace AIT {

	/// @brief A node of a decision tree.
	template <typename TSplitPoint, typename TStatistics>
	class Node {
		TSplitPoint split_point_;
		TStatistics statistics_;

	public:
		typedef std::vector<int>::size_type size_type;

		virtual Direction Evaluate(const DataPointCollection<> &data_point_collection, size_type index) const = 0;

		const TSplitPoint & GetSplitPoint() const {
			return split_point_;
		}

		void SetSplitPoint(const TSplitPoint &split_point) {
			split_point_ = split_point;
		}

		const TStatistics & GetStatistics() const {
			return statistics_;
		}

		void SetStatistics(const TStatistics &statistics) {
			statistics_ = statistics;
		}

	};

}

#endif
