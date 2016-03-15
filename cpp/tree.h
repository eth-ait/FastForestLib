//
//  tree.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

#include <vector>
#include <memory>
#include <iostream>

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/utility/enable_if.hpp>
#ifdef SERIALIZE_WITH_BOOST
#include <boost/serialization/vector.hpp>
#endif
#include <cereal/types/vector.hpp>


#include "ait.h"
#include "node.h"
#include "mpl_utils.h"

namespace ait {

/// @brief A decision tree.
template <typename TSplitPoint, typename TStatistics>
class Tree
{
public:
    using NodeT = Node<TSplitPoint, TStatistics>;

    struct NodeEntry
    {
        NodeT node;
        bool is_leaf;

        NodeEntry() : is_leaf(false)
        {}

private:
#ifdef SERIALIZE_WITH_BOOST
        friend class boost::serialization::access;
        
        template <typename Archive>
        void serialize(Archive& archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
        {
            archive & node;
            archive & is_leaf;
        }
#endif
        
        friend class cereal::access;
        
        template <typename Archive>
        void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
        {
            archive(cereal::make_nvp("node", node));
            archive(cereal::make_nvp("is_leaf", is_leaf));
        }
    };

    template <typename BaseIterator, typename ValueType>
    class NodeIterator_ : public boost::iterator_adaptor<NodeIterator_<BaseIterator, ValueType>, BaseIterator, ValueType>
    {
    public:
        explicit NodeIterator_(BaseIterator it, BaseIterator begin)
        : NodeIterator_::iterator_adaptor_(it), begin_(begin)
        {}

        template <typename OtherBaseIterator, typename OtherValueType>
        NodeIterator_(
                               const NodeIterator_<OtherBaseIterator, OtherValueType>& other,
                               typename boost::enable_if<
                               boost::is_convertible<OtherBaseIterator, BaseIterator>, int>::type = 0
                               )
        : NodeIterator_::iterator_adaptor_(other.base()), begin_(other.begin_)
        {}

        bool is_root() const
        {
            return this->base() == begin_;
        }
        
        bool is_leaf() const
        {
            return this->base()->is_leaf;
        }
        
        void set_leaf(bool is_leaf = true)
        {
            this->base()->is_leaf = is_leaf;
        }

        size_type get_node_index() const
        {
            return this->base() - begin_;
        }

        void goto_left_child()
        {
            assert(!this->is_leaf());
            size_type left_child_offset = get_node_index() + 1;
            this->base_reference() += left_child_offset;
        }
        
        void goto_right_child()
        {
            assert(!this->is_leaf());
            size_type right_child_offset = get_node_index() + 2;
            this->base_reference() += right_child_offset;
        }
        
        void goto_parent()
        {
            assert(!this->is_root());
            size_type parent_offset = - (get_node_index() - 1) / 2;
            this->base_reference() += parent_offset;
        }
        
        NodeIterator_ left_child() const
        {
            // TODO: Remove assertion or necessary for staying within the tree
//            assert(!this->is_leaf());
            size_type left_child_offset = get_node_index() + 1;
            return NodeIterator_(this->base() + left_child_offset, begin_);
        }
        
        NodeIterator_ right_child() const
        {
            // TODO: Remove assertion or necessary for staying within the tree
//            assert(!this->is_leaf());
            size_type right_child_offset = get_node_index() + 2;
            return NodeIterator_(this->base() + right_child_offset, begin_);
        }
        
        NodeIterator_ parent() const
        {
            assert(!this->is_root());
            size_type parent_offset = - (get_node_index() - 1) / 2;
            return NodeIterator_(this->base() + parent_offset, begin_);
        }

    private:
		using IteratorAdapterType = boost::iterator_adaptor<NodeIterator_<BaseIterator, ValueType>, BaseIterator, ValueType>;

        BaseIterator begin_;

        friend class boost::iterator_core_access;
        template <typename, typename> friend class NodeIterator_;

        typename IteratorAdapterType::iterator_facade_::reference dereference() const
        {
            return this->base()->node;
        }

    };

public:
    using NodeIterator = NodeIterator_<typename std::vector<NodeEntry>::iterator, NodeT>;
    using ConstNodeIterator = NodeIterator_<typename std::vector<NodeEntry>::const_iterator, const NodeT>;

    class TreeLevel
    {
    protected:
        Tree& tree_;
        size_type level_;

    public:
        using iterator = NodeIterator;
        using const_iterator = ConstNodeIterator;

        TreeLevel(Tree& tree, size_type level)
        : tree_(tree), level_(level)
        {}

        iterator begin()
        {
            size_type offset = compute_node_offset(level_);
            return tree_.begin() + offset;
        }

        iterator end()
        {
            size_type offset = compute_node_offset(level_ + 1);
            return tree_.begin() + offset;
        }
        
        const_iterator cbegin() const
        {
            size_type offset = compute_node_offset(level_);
            return tree_.cbegin() + offset;
        }
        
        const_iterator cend() const
        {
            size_type offset = compute_node_offset(level_ + 1);
            return tree_.cbegin() + offset;
        }

        size_type size() const
        {
            return compute_node_offset(level_ + 1) - compute_node_offset(level_);
        }

    protected:
        size_type compute_node_offset(size_type level) const
        {
            size_type offset = (1 << (level - 1)) - 1;
            return offset;
        }
    };

    /// @brief Create an empty tree with no nodes.
    Tree()
    : depth_(0)
    {}

    /// @brief Create a tree.
    /// @param depth The depth of the tree. A depth of 1 corresponds to a tree
    ///              with a single root node.
    Tree(size_type depth)
    : depth_(depth)
    {
        size_type num_of_nodes = (1 << depth_) - 1;
        node_entries_.resize(num_of_nodes);
        size_type first_leaf_node_index = (1 << (depth - 1)) - 1;
        for (size_type i = first_leaf_node_index; i < num_of_nodes; i++)
        {
            node_entries_[i].is_leaf = true;
        }
    }
    
    /// @brief Return the node iterator pointing to the root.
    NodeIterator begin()
    {
        return NodeIterator(node_entries_.begin(), node_entries_.begin());
    }
    
    /// @brief Return the node iterator pointing to the last node.
    NodeIterator end()
    {
        return NodeIterator(node_entries_.end(), node_entries_.begin());
    }

    /// @brief Return the constant node iterator pointing to the root.
    ConstNodeIterator cbegin() const
    {
        return ConstNodeIterator(node_entries_.cbegin(), node_entries_.cbegin());
    }
    
    /// @brief Return the constant node iterator pointing to the last node.
    ConstNodeIterator cend() const
    {
        return ConstNodeIterator(node_entries_.cend(), node_entries_.cbegin());
    }

    /// @brief Return the node iterator pointing to the root.
    NodeIterator get_root_iterator()
    {
        return begin();
    }
    
    /// @brief Return a node in the tree.
    ConstNodeIterator get_root_iterator() const
    {
        return cbegin();
    }
    
    /// @brief Return a node entry.
    NodeEntry& get_node(size_type index)
    {
        return node_entries_[index];
    }
    
    /// @brief Return a const node entry.
    const NodeEntry& get_node(size_type index) const
    {
        return node_entries_[index];
    }

    /// @brief Return an iterator pointing to a node in the tree.
    NodeIterator get_node_iterator(size_type index)
    {
        return begin() + index;
    }
    
    /// @brief Return an iterator pointing to a node in the tree.
    ConstNodeIterator get_node_iterator(size_type index) const
    {
        return cbegin() + index;
    }
    
    /// @brief Return depth of the tree. A depth of 1 corresponds to a tree
    ///        with a single node.
    size_type depth() const
    {
        return depth_;
    }

    /// @brief Return number of nodes in the tree.
    size_type size() const
    {
        return node_entries_.size();
    }

    /// @brief evaluate a collection of data-points on the tree.
    /// @param data The collection of data-points
    /// @param leaf_node_indices A vector for storing the results. For each
    ///                          data-point it will contain the index of the
    ///                          corresponding leaf node.
    template <typename TSample>
    void evaluate(const std::vector<TSample>& samples,
        std::vector<size_type>& leaf_node_indices) const
    {
        leaf_node_indices.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            leaf_node_indices.push_back(node_iter.get_index());
        }
    }
    
    /// @brief evaluate a collection of data-points on the tree.
    /// @param data The collection of data-points
    /// @param leaf_node_indices A vector for storing the results. For each
    ///                          data-point it will contain the index of the
    ///                          corresponding leaf node.
    template <typename TSampleIterator>
    void evaluate(const TSampleIterator& it_start, const TSampleIterator& it_end, std::vector<size_type>& leaf_node_indices) const
    {
        leaf_node_indices.reserve(it_end - it_start);
        for (auto it = it_start; it != it_end; it++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            leaf_node_indices.push_back(node_iter.get_node_index());
        }
    }

    /// @brief evaluate a collection of data-points on the tree and return the iterators (const version).
    /// @param data The collection of data-points
    /// @param max_depth The maximum depth to which the tree should be traversed
    template <typename TSample>
    const std::vector<ConstNodeIterator> evaluate(const std::vector<TSample>& samples, size_type max_depth = std::numeric_limits<size_type>::max()) const
    {
        std::vector<ConstNodeIterator> leaf_nodes;
        leaf_nodes.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstNodeIterator node_iter = evaluate(*it, max_depth);
            leaf_nodes.push_back(std::move(node_iter));
        }
        return leaf_nodes;
    }
    
    /// @brief evaluate a collection of data-points on the tree and return the iterators.
    /// @param data The collection of data-points
    /// @param max_depth The maximum depth to which the tree should be traversed
    template <typename TSample>
    std::vector<NodeIterator> evaluate(const std::vector<TSample>& samples, size_type max_depth = std::numeric_limits<size_type>::max())
    {
        std::vector<NodeIterator> leaf_nodes;
        leaf_nodes.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            NodeIterator node_iter = evaluate(*it, max_depth);
            leaf_nodes.push_back(std::move(node_iter));
        }
        return leaf_nodes;
    }

    /// @brief Evaluate a data-points on the tree (const version).
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename TSample>
    ConstNodeIterator evaluate(const TSample& sample, size_type max_depth = std::numeric_limits<size_type>::max()) const
    {
        size_type current_depth = 1;
        ConstNodeIterator node_iter = get_root_iterator();
        while (!node_iter.is_leaf() && current_depth < max_depth)
        {
            Direction direction = node_iter->get_split_point().evaluate(sample);
            if (direction == Direction::LEFT) {
                node_iter.goto_left_child();
            } else {
                node_iter.goto_right_child();
            }
            ++current_depth;
        }
        return node_iter;
    }
    
    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename TSample>
    NodeIterator evaluate(const TSample& sample, size_type max_depth = std::numeric_limits<size_type>::max())
    {
        size_type current_depth = 1;
        NodeIterator node_iter = get_root_iterator();
        while (!node_iter.is_leaf() && current_depth < max_depth)
        {
            Direction direction = node_iter->get_split_point().evaluate(sample);
            if (direction == Direction::LEFT) {
                node_iter.goto_left_child();
            } else {
                node_iter.goto_right_child();
            }
            ++current_depth;
        }
        return node_iter;
    }

    template <typename TSample>
    void evaluate_parallel(const std::vector<TSample>& samples, std::function<void(const TSample& , ConstNodeIterator&)>& func) const
    {
        //#pragma omp parallel for
        for (size_type i = 0; i < samples.size(); i++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(samples[i]);
            func(samples[i], node_iter);
        }
    }

    template <typename TSample>
    void evaluate_parallel(const std::vector<TSample>& samples, const std::function<void(const TSample& , const NodeT&)>& func) const
    {
        std::function<void(const TSample& , ConstNodeIterator& )> func_wrapper = [&func](const TSample& sample, ConstNodeIterator& node_iter)
        {
            func(sample, *node_iter);
        };
        evaluate_parallel(samples, func_wrapper);
    }

    template <typename TSample>
    void evaluate(const std::vector<TSample>& samples, std::function<void(const TSample& , ConstNodeIterator&)>& func) const
    {
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            func(*it, node_iter);
        }
    }

    template <typename TSample>
    void evaluate(const std::vector<TSample>& samples, const std::function<void (const TSample& , const NodeT&)>& func) const
    {
        std::function<void (const TSample& , ConstNodeIterator& )> func_wrapper = [&func] (const TSample& sample, ConstNodeIterator& node_iter)
        {
            func(sample, *node_iter);
        };
        evaluate(samples, func_wrapper);
    }
    
    template <typename TSampleIterator>
    void evaluate(const TSampleIterator& it_start, const TSampleIterator& it_end, std::function<void(const TSampleIterator& , ConstNodeIterator&)>& func) const
    {
        for (auto it = it_start; it != it_end; it++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            func(it, node_iter);
        }
    }
    
    template <typename TSampleIterator>
    void evaluate(const TSampleIterator& it_start, const TSampleIterator& it_end, const std::function<void (const TSampleIterator& , const NodeT&)>& func) const
    {
        std::function<void (const TSampleIterator& , ConstNodeIterator& )> func_wrapper = [&func] (const TSampleIterator& sample_it, ConstNodeIterator& node_iter)
        {
            func(sample_it, *node_iter);
        };
        evaluate(it_start, it_end, func_wrapper);
    }

    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename TSample>
    ConstNodeIterator evaluate_to_iterator(const TSample& sample) const
    {
        ConstNodeIterator node_iter = get_root_iterator();
        while (!node_iter.is_leaf())
        {
            Direction direction = node_iter->get_split_point().evaluate(sample);
            if (direction == Direction::LEFT)
            {
                node_iter.goto_left_child();
            }
            else
            {
                node_iter.goto_right_child();
            }
        }
        return node_iter;
    }

private:
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive & depth_;
        archive & node_entries_;
    }
#endif
    
    friend class cereal::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("depth", depth_));
        archive(cereal::make_nvp("node_entries", node_entries_));
    }

    size_type depth_;
    std::vector<NodeEntry> node_entries_;
};

}
