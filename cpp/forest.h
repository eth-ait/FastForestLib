#ifndef AITDistributedRandomForest_forest_h
#define AITDistributedRandomForest_forest_h

#include <memory>
#include <vector>
#include <functional>

#include "tree.h"

namespace AIT {

	/// @brief A decision forest, i.e. a collection of decision trees.
	template <typename TSplitPoint, typename TStatistics>
	class Forest {
	public:
		typedef Tree<TSplitPoint, TStatistics> TreeType;
        typedef typename TreeType::NodeType NodeType;
		typedef std::vector<int>::size_type size_type;

	private:
		std::vector<TreeType> trees_;

	public:

		/// @brief Create an empty forest.
		Forest() {}

		/// @brief Add a tree to the forest.
		void AddTree(const TreeType &tree)
		{
			trees_.push_back(tree);
		}
        
        /// @brief Add a tree to the forest.
        void AddTree(TreeType &&tree)
        {
            trees_.push_back(std::move(tree));
        }
        

		/// @brief Return a tree in the forest.
		const TreeType & GetTree(size_type index) const
		{
			return trees_[index];
		}

		/// @brief Return a tree in the forest.
		TreeType & GetTree(size_type index)
		{
			return trees_[index];
		}

		/// @brief Return number of trees in the forest.
		size_type NumOfTrees() const
		{
			return trees_.size();
		}

        // TODO: Think about evaluation methods
        template <typename Sample>
        void Evaluate(const std::vector<Sample> &samples, const std::function<void (const typename TreeType::ConstNodeIterator &)> &func) const {
            for (size_type i=0; i < NumOfTrees(); i++) {
                const TreeType &tree = trees_[i];
                for (auto it = samples.cbegin(); it != samples.cend(); it++) {
                    typename TreeType::ConstNodeIterator node_iter = tree.EvaluateToIterator(*it);
                    func(node_iter);
                }
            }
        }

        template <typename Sample>
        void Evaluate(const std::vector<Sample> &samples, const std::function<void (const NodeType &)> &func) const {
            Evaluate(samples, [&func] (const typename TreeType::ConstNodeIterator &node_iter) {
//                func(*node_iter);
            });
        }

		/// @brief Evaluate a collection of data-points on each tree in the forest.
		/// @param data The collection of data-points
        /// @return A vector of the results. See #Evaluate().
        template <typename Sample>
        std::vector<std::vector<size_type> > Evaluate(const std::vector<Sample> &samples) const
		{
            std::vector<std::vector<size_type> > forest_leaf_node_indices;
            Evaluate(samples, forest_leaf_node_indices);
			return forest_leaf_node_indices;
		}

		/// @brief Evaluate a collection of data-points on each tree in the forest.
		/// @param data The collection of data-points
		/// @param forest_leaf_node_indices A vector for storing the results. For each tree
		///                          it will contain another vector storing the
		///                          index of the corresponding leaf node for each
		///                          data-point.
		///                          So leaf_node_indices.size() will be equal to
        ///                          NumOfTrees().
        template <typename Sample>
        void Evaluate(const std::vector<Sample> &samples, std::vector<std::vector<size_type> > &forest_leaf_node_indices) const
        {
            forest_leaf_node_indices.reserve(NumOfTrees());
			for (size_type i=0; i < NumOfTrees(); i++) {
                std::vector<size_type> leaf_node_indices;
                leaf_node_indices.reserve(samples.size());
                trees_[i].Evaluate(samples, leaf_node_indices);
                forest_leaf_node_indices.push_back(std::move(leaf_node_indices));
			}
		}
        
        template <typename Sample>
        const std::vector<std::vector<typename TreeType::ConstNodeIterator> > EvaluateToIterators(const std::vector<Sample> &samples) const
        {
            std::vector<std::vector<typename TreeType::ConstNodeIterator> > forest_leaf_nodes;
            forest_leaf_nodes.reserve(NumOfTrees());
            for (size_type i=0; i < NumOfTrees(); i++) {
                std::vector<typename TreeType::ConstNodeIterator> leaf_nodes;
                leaf_nodes.reserve(samples.size());
                for (auto it = samples.cbegin(); it != samples.cend(); it++) {
                    typename TreeType::ConstNodeIterator node_iter = Evaluate(*it);
                    leaf_nodes.push_back(node_iter);
                }
                forest_leaf_nodes.push_back(std::move(leaf_nodes));
            }
            return forest_leaf_nodes;
        }
        

	};

}

#endif
