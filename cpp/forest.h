#ifndef AITDistributedRandomForest_forest_h
#define AITDistributedRandomForest_forest_h

#include <memory>
#include <vector>

#include "tree.h"

namespace AIT {

	/// @brief A decision forest, i.e. a collection of decision trees.
	template <typename TSplitPoint, typename TStatistics>
	class Forest {
	public:
		typedef Tree<TSplitPoint, TStatistics> tree_type;
		typedef std::vector<int>::size_type size_type;

	private:
		std::vector<std::unique_ptr<tree_type>> trees_;

	public:

		/// @brief Create an empty forest.
		Forest() {}

		/// @brief Add a tree to the forest.
		void AddTree(std::unique_ptr<tree_type> tree)
		{
			trees_.push_back(std::move(tree));
		}

		/// @brief Return a tree in the forest.
		const std::unique_ptr<tree_type>& GetTree(size_type index) const
		{
			return trees_[index];
		}

		/// @brief Return a tree in the forest.
		std::unique_ptr<tree_type>& GetTree(size_type index)
		{
			return trees_[index];
		}

		/// @brief Return number of trees in the forest.
		size_type NumOfTrees() const
		{
			return trees_.size();
		}

		/// @brief Evaluate a collection of data-points on each tree in the forest.
		/// @param data The collection of data-points
		/// @return A vector of the results. See #Evaluate().
		std::vector<std::vector<size_type>> Evaluate(const DataPointCollection<> &data_point_collection) const
		{
			std::vector<std::vector<size_type>> leaf_node_indices;
			Evaluate(data_point_collection, leaf_node_indices);
			return leaf_node_indices;
		}

		/// @brief Evaluate a collection of data-points on each tree in the forest.
		/// @param data The collection of data-points
		/// @param leaf_node_indices A vector for storing the results. For each tree
		///                          it will contain another vector storing the
		///                          index of the corresponding leaf node for each
		///                          data-point.
		///                          So leaf_node_indices.size() will be equal to
		///                          NumOfTrees().
		void Evaluate(const DataPointCollection<> &data_point_collection, std::vector<std::vector<size_type>> &leaf_node_indices) const
		{
			leaf_node_indices.resize(NumOfTrees());
			for (size_type i=0; i < NumOfTrees(); i++) {
			leaf_node_indices[i].resize(data_point_collection.Count());
			trees_[i]->Evaluate(data_point_collection, leaf_node_indices[i]);
			}
		}

	};

}

#endif
