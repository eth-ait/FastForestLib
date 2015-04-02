#ifndef AITDistributedRandomForest_tree_h
#define AITDistributedRandomForest_tree_h

#include <vector>
#include <memory>
#include <iostream>

#include "node.h"

namespace AIT {

	/// @brief A decision tree.
	template <typename TSplitPoint, typename TStatistics>
	class Tree {
	public:
		typedef Node<TSplitPoint, TStatistics> NodeType;
		typedef std::vector<int>::size_type size_type;

	private:
		const size_type depth_;
		const size_type first_leaf_node_index_;
		std::vector<NodeType> nodes_;

        template <typename TreeType, typename ValueType>
        class NodeIterator_ {
        protected:
            TreeType *tree_ptr_;
            size_type node_index_;

        public:
            NodeIterator_(TreeType *tree_ptr, size_type node_index)
            : tree_ptr_(tree_ptr), node_index_(node_index)
            {}

            ValueType & operator*()
            {
                return tree_ptr_->nodes_[node_index_];
            }

            ValueType * operator->()
            {
                return &tree_ptr_->nodes_[node_index_];
            }

//            bool operator==(const NodeType &other) const
//            {
//                return node_index_ == other.node_index_;
//            }
//
//            bool operator!=(const NodeType &other) const
//            {
//                return !this->operator==(other);
//            }

            size_type GetNodeIndex() const
            {
                return node_index_;
            }

            bool IsLeafNode() const
            {
                return node_index_ >= tree_ptr_->first_leaf_node_index_;
            }

            NodeIterator_ LeftChild()
            {
                size_type left_child_index = 2 * node_index_ + 1;
                return NodeIterator_(tree_ptr_, left_child_index);
            }
            
            NodeIterator_ RightChild()
            {
                size_type right_child_index = 2 * node_index_ + 2;
                return NodeIterator_(tree_ptr_, right_child_index);
            }

            NodeIterator_ Parent()
            {
                size_type parent_index = (node_index_ - 1) / 2;
                return NodeIterator_(tree_ptr_, parent_index);
            }

        };

    public:
        typedef NodeIterator_<Tree<TSplitPoint, TStatistics>, NodeType> NodeIterator;
        typedef NodeIterator_<Tree<TSplitPoint, TStatistics> const, NodeType const> ConstNodeIterator;

		/// @brief Create a tree.
		/// @param depth The depth of the tree. A depth of 0 corresponds to a tree
		///              with a single node.
		Tree(size_type depth)
        : depth_(depth), first_leaf_node_index_((1 << depth) - 1)
        {
			size_type num_of_nodes = (1 << (depth_ + 1)) - 1;
			nodes_.resize(num_of_nodes);
		}

        /// @brief Return a node in the tree.
        NodeIterator GetRoot()
        {
            return NodeIterator(this, 0);
        }
        
        /// @brief Return a node in the tree.
        ConstNodeIterator GetRoot() const
        {
            return ConstNodeIterator(this, 0);
        }

		/// @brief Return a node in the tree.
		NodeIterator GetNode(size_type index)
		{
			return NodeIterator(this, index);
        }
        
        /// @brief Return a node in the tree.
        ConstNodeIterator GetNode(size_type index) const
        {
            return ConstNodeIterator(this, index);
        }
        
		/// @brief Return depth of the tree. A depth of 0 corresponds to a tree
		///        with a single node.
		size_type Depth() const
        {
			return depth_;
		}

		/// @brief Return number of nodes in the tree.
		size_type NumOfNodes() const
		{
			return nodes_.size();
		}

		/// @brief Evaluate a collection of data-points on the tree.
		/// @param data The collection of data-points
		/// @param leaf_node_indices A vector for storing the results. For each
		///                          data-point it will contain the index of the
        ///                          corresponding leaf node.
        template <typename Sample>
        void Evaluate(const std::vector<Sample> &samples,
			std::vector<size_type> &leaf_node_indices) const
		{
            leaf_node_indices.reserve(samples.size());
            for (auto it = samples.cbegin(); it != samples.cend(); it++) {
                ConstNodeIterator node_iter = EvaluateToIterator(*it);
                leaf_node_indices.push_back(node_iter.GetNodeIndex());
			}
		}
        
        template <typename Sample>
        const std::vector<ConstNodeIterator> Evaluate(const std::vector<Sample> &samples) const
        {
            std::vector<ConstNodeIterator> leaf_nodes;
            leaf_nodes.reserve(samples.size());
            for (auto it = samples.cbegin(); it != samples.cend(); it++) {
                ConstNodeIterator node_iter = Evaluate(*it);
                leaf_nodes.push_back(node_iter);
            }
            return leaf_nodes;
        }
        
        /// @brief Evaluate a data-points on the tree.
        /// @param data_point The data-point.
        /// @return The index of the corresponding leaf-node.
        template <typename Sample>
        size_type Evaluate(const Sample &sample) const {
            ConstNodeIterator node_iter = GetRoot();
            while (!node_iter.IsLeafNode()) {
                Direction direction = node_iter->GetSplitPoint().Evaluate(sample);
                if (direction == Direction::LEFT)
                    node_iter = node_iter.LeftChild();
                else
                    node_iter = node_iter.RightChild();
            }
            return node_iter.GetNodeIndex();
        }

		/// @brief Evaluate a data-points on the tree.
		/// @param data_point The data-point.
		/// @return The index of the corresponding leaf-node.
        template <typename Sample>
		ConstNodeIterator EvaluateToIterator(const Sample &sample) const {
            ConstNodeIterator node_iter = GetRoot();
			while (!node_iter.IsLeafNode()) {
                Direction direction = node_iter->GetSplitPoint().Evaluate(sample);
                if (direction == Direction::LEFT)
                    node_iter = node_iter.LeftChild();
                else
                    node_iter = node_iter.RightChild();
			}
			return node_iter;
		}

	private:
		// @brief Return the left child index of the specified node.
		size_type GetLeftChildIndex(size_type index) const
        {
			return 2 * index + 1;
		}

		// @brief Return the right child index of the specified node.
		size_type GetRightChildIndex(size_type index) const
        {
			return 2 * index + 2;
		}

		// @brief Check if the specified node is a leaf node.
		bool IsLeafNodeIndex(size_type index) const
        {
			if (index >= first_leaf_node_index_)
				return true;
			else
				return false;
		}

	};

}

#endif
