#ifndef AITDistributedRandomForest_node_h
#define AITDistributedRandomForest_node_h

#include <memory>
#include <vector>


namespace AIT {
    
    enum class Direction { LEFT = -1, RIGHT = +1 };

	/// @brief A node of a decision tree.
	template <typename TSplitPoint, typename TStatistics>
	class Node {
		TSplitPoint split_point_;
		TStatistics statistics_;

	public:
		typedef std::vector<int>::size_type size_type;

		~Node() {}

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
        
        template <typename Archive>
        void serialize(Archive &archive, const unsigned int version)
        {
            archive(cereal::make_nvp("split_point", split_point_));
            archive(cereal::make_nvp("statistics", statistics_));
        }

	};

}

#endif
