#pragma once

#include <vector>
#include <memory>
#include <iostream>

#include <cereal/types/vector.hpp>

#include "ait.h"
#include "node.h"

// TODO: Modify iterator interface to reflect STL usage.
namespace ait {

/// @brief A decision tree.
template <typename TSplitPoint, typename TStatistics>
class Tree
{
public:
    using NodeType = Node<TSplitPoint, TStatistics>;

private:
    struct NodeEntry
    {
        NodeType node;
        bool is_leaf;
        
        NodeEntry() : is_leaf(false)
        {}
        
        template <typename Archive>
        void serialize(Archive &archive, const unsigned int version)
        {
            archive(cereal::make_nvp("node", node));
            archive(cereal::make_nvp("is_leaf", is_leaf));
        }
    };

    template <typename TreeType, typename ValueType>
    class TreeIterator_
    {
    protected:
        TreeType &tree_;
        size_type node_index_;

    public:
        TreeIterator_(TreeType &tree, size_type node_index)
        : tree_(tree), node_index_(node_index)
        {}

        ValueType & operator*()
        {
            return tree_.node_entries_[node_index_].node;
        }
        
        ValueType * operator->()
        {
            return &tree_.node_entries_[node_index_].node;
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
        
        size_type get_index() const
        {
            return node_index_;
        }
        
        bool is_root() const
        {
            return node_index_ == 0;
        }

        bool is_leaf() const
        {
            return tree_.node_entries_[node_index_].is_leaf;
        }
        
        void set_leaf(bool is_leaf = true)
        {
            tree_.node_entries_[node_index_].is_leaf = is_leaf;
        }
        
        void goto_left_child()
        {
            assert(!this->is_leaf());
            size_type left_child_index = 2 * node_index_ + 1;
            node_index_ = left_child_index;
        }
        
        void goto_right_child()
        {
            assert(!this->is_leaf());
            size_type right_child_index = 2 * node_index_ + 2;
            node_index_ = right_child_index;
        }
        
        void goto_parent()
        {
            assert(!this->is_root_node());
            size_type parent_index = (node_index_ - 1) / 2;
            node_index_ = parent_index;
        }
        
        TreeIterator_ left_child()
        {
            assert(!this->is_leaf());
            size_type left_child_index = 2 * node_index_ + 1;
            return TreeIterator_(tree_, left_child_index);
        }
        
        TreeIterator_ right_child()
        {
            assert(!this->is_leaf());
            size_type right_child_index = 2 * node_index_ + 2;
            return TreeIterator_(tree_, right_child_index);
        }
        
        TreeIterator_ parent()
        {
            assert(!this->is_root_node());
            size_type parent_index = (node_index_ - 1) / 2;
            return TreeIterator_(tree_, parent_index);
        }
    };

    size_type depth_;
    std::vector<NodeEntry> node_entries_;

    // TODO: Use boost::iterator to implement a proper iterator
    class NodeEntryIteratorWrapper
    {
    public:
        using BaseIterator = typename std::vector<NodeEntry>::iterator;

    private:
        BaseIterator it_;
        
    public:
        NodeEntryIteratorWrapper(const BaseIterator &it)
        : it_(it)
        {}
        
        NodeType & operator*()
        {
            return it_->node;
        }
        
        NodeType * operator->()
        {
            return &it_->node;
        }

        bool operator==(const NodeEntryIteratorWrapper &other) const
        {
            return it_ == other.it_;
        }

        bool operator!=(const NodeEntryIteratorWrapper &other) const
        {
            return it_ != other.it_;
        }
        
        NodeEntryIteratorWrapper & operator++()
        {
            ++it_;
            return *this;
        }
        
        NodeEntryIteratorWrapper operator++(int)
        {
            NodeEntryIteratorWrapper tmp(*this);
            ++(*this);
            return tmp;
        }
    };
    
public:
    using TreeIterator = TreeIterator_<Tree<TSplitPoint, TStatistics>, NodeType>;
    using ConstTreeIterator = TreeIterator_<Tree<TSplitPoint, TStatistics> const, NodeType const>;
    
    class TreeLevel
    {
    protected:
        Tree &tree_;
        size_type level_;
        
    public:
        TreeLevel(Tree &tree, size_type level)
        : tree_(tree), level_(level)
        {}
        
        NodeEntryIteratorWrapper begin()
        {
            size_type offset = (1 << (level_ - 1)) - 1;
            return NodeEntryIteratorWrapper(tree_.node_entries_.begin() + offset);
        }

        NodeEntryIteratorWrapper end()
        {
            size_type offset = (1 << level_) - 1;
            return NodeEntryIteratorWrapper(tree_.node_entries_.begin() + offset);
        }
    };

    /// @brief Create an empty tree with no nodes.
    Tree()
    : depth_(0)
    {}

    /// @brief Create a tree.
    /// @param depth The depth of the tree. A depth of 1 corresponds to a tree
    ///              with a single node.
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

    /// @brief Return a node in the tree.
    TreeIterator get_root()
    {
        return TreeIterator(*this, 0);
    }
    
    /// @brief Return a node in the tree.
    ConstTreeIterator get_root() const
    {
        return ConstTreeIterator(*this, 0);
    }

    /// @brief Return a node in the tree.
    TreeIterator get_node(size_type index)
    {
        return TreeIterator(*this, index);
    }
    
    /// @brief Return a node in the tree.
    ConstTreeIterator get_node(size_type index) const
    {
        return ConstTreeIterator(*this, index);
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
    template <typename Sample>
    void evaluate(const std::vector<Sample> &samples,
        std::vector<size_type> &leaf_node_indices) const
    {
        leaf_node_indices.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstTreeIterator node_iter = evaluate_to_iterator(*it);
            leaf_node_indices.push_back(node_iter.get_index());
        }
    }
    
    /// @brief evaluate a collection of data-points on the tree and return the iterators (const version).
    /// @param data The collection of data-points
    /// @param max_depth The maximum depth to which the tree should be traversed
    template <typename Sample>
    const std::vector<ConstTreeIterator> evaluate(const std::vector<Sample> &samples, size_type max_depth = std::numeric_limits<size_type>::max()) const
    {
        std::vector<ConstTreeIterator> leaf_nodes;
        leaf_nodes.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstTreeIterator node_iter = evaluate(*it, max_depth);
            leaf_nodes.push_back(std::move(node_iter));
        }
        return leaf_nodes;
    }
    
    /// @brief evaluate a collection of data-points on the tree and return the iterators.
    /// @param data The collection of data-points
    /// @param max_depth The maximum depth to which the tree should be traversed
    template <typename Sample>
    std::vector<TreeIterator> evaluate(const std::vector<Sample> &samples, size_type max_depth = std::numeric_limits<size_type>::max())
    {
        std::vector<TreeIterator> leaf_nodes;
        leaf_nodes.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            TreeIterator node_iter = evaluate(*it, max_depth);
            leaf_nodes.push_back(std::move(node_iter));
        }
        return leaf_nodes;
    }

    template <typename Sample>
    std::vector<ConstTreeIterator> evaluate(const std::vector<Sample> &samples, size_type max_depth = std::numeric_limits<size_type>::max()) const
    {
        std::vector<ConstTreeIterator> leaf_nodes;
        leaf_nodes.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstTreeIterator node_iter = evaluate(*it, max_depth);
            leaf_nodes.push_back(std::move(node_iter));
        }
        return leaf_nodes;
    }
    

    /// @brief Evaluate a data-points on the tree (const version).
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename Sample>
    ConstTreeIterator evaluate(const Sample &sample, size_type max_depth = std::numeric_limits<size_type>::max()) const
    {
        size_type current_depth = 1;
        ConstTreeIterator node_iter = get_root();
        while (!node_iter.is_leaf() && current_depth < max_depth)
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
            ++current_depth;
        }
        return node_iter;
    }
    
    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename Sample>
    TreeIterator evaluate(const Sample &sample, size_type max_depth = std::numeric_limits<size_type>::max())
    {
        size_type current_depth = 1;
        TreeIterator node_iter = get_root();
        while (!node_iter.is_leaf() && current_depth < max_depth)
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
            ++current_depth;
        }
        return node_iter;
    }

    template <typename Sample>
    void evaluate_parallel(const std::vector<Sample> &samples, std::function<void(const Sample &, ConstTreeIterator &)> &func) const
    {
        //#pragma omp parallel for
        for (int i = 0; i < samples.size(); i++)
        {
            ConstTreeIterator node_iter = evaluate_to_iterator(samples[i]);
            func(samples[i], node_iter);
        }
    }

    template <typename Sample>
    void evaluate_parallel(const std::vector<Sample> &samples, const std::function<void(const Sample &, const NodeType &)> &func) const
    {
        std::function<void(const Sample &, ConstTreeIterator &)> func_wrapper = [&func](const Sample &sample, ConstTreeIterator &node_iter)
        {
            func(sample, *node_iter);
        };
        evaluate_parallel(samples, func_wrapper);
    }

    template <typename Sample>
    void evaluate(const std::vector<Sample> &samples, std::function<void(const Sample &, ConstTreeIterator &)> &func) const
    {
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstTreeIterator node_iter = evaluate_to_iterator(*it);
            func(*it, node_iter);
        }
    }

    template <typename Sample>
    void evaluate(const std::vector<Sample> &samples, const std::function<void (const Sample &, const NodeType &)> &func) const
    {
        std::function<void (const Sample &, ConstTreeIterator &)> func_wrapper = [&func] (const Sample &sample, ConstTreeIterator &node_iter)
        {
            func(sample, *node_iter);
        };
        evaluate(samples, func_wrapper);
    }

    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename Sample>
    ConstTreeIterator evaluate_to_iterator(const Sample &sample) const
    {
        ConstTreeIterator node_iter = get_root();
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

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version)
    {
        archive(cereal::make_nvp("depth", depth_));
        archive(cereal::make_nvp("node_entries", node_entries_));
    }

private:
    // @brief Return the left child index of the specified node.
    size_type get_left_child_index(size_type index) const
    {
        return 2 * index + 1;
    }

    // @brief Return the right child index of the specified node.
    size_type get_right_child_index(size_type index) const
    {
        return 2 * index + 2;
    }
};

template <typename TSplitPoint, typename TStatistics, typename TSample, typename TMatrix = Eigen::MatrixXd>
class TreeUtilities
{
    using TreeType = Tree<TSplitPoint, TStatistics>;

    const TreeType &tree_;

public:
    TreeUtilities(const TreeType &tree)
    : tree_(tree)
    {}
    
    // TODO: Samples should contain information on number of labels
    template <int num_of_labels>
    TMatrix compute_confusion_matrix(const std::vector<TSample> &samples) const
    {
        TMatrix confusion_matrix(num_of_labels, num_of_labels);
        confusion_matrix.setZero();
        /*tree_.template evaluate<TSample>(samples, [&confusion_matrix] (const TSample &sample, const typename TreeType::NodeType &node) {
                            typename TSample::label_type true_label = sample.GetLabel();
                            const TStatistics &statistics = node.GetStatistics();
                            const auto &histogram = statistics.GetHistogram();
                            typename TSample::label_type predicted_label = std::max_element(histogram.cbegin(), histogram.cend()) - histogram.cbegin();
                            confusion_matrix(true_label, predicted_label)++;
        });*/
        tree_.template evaluate_parallel<TSample>(samples, [&confusion_matrix](const TSample &sample, const typename TreeType::NodeType &node)
        {
            size_type true_label = sample.get_label();
            const TStatistics &statistics = node.get_statistics();
            const auto &histogram = statistics.get_histogram();
            size_type predicted_label = std::max_element(histogram.cbegin(), histogram.cend()) - histogram.cbegin();
            confusion_matrix(true_label, predicted_label)++;
        });
        return confusion_matrix;
    }
    
    template <int num_of_labels>
    TMatrix compute_normalized_confusion_matrix(const std::vector<TSample> &samples) const
    {
        TMatrix confusion_matrix = compute_confusion_matrix<num_of_labels>(samples);
        auto row_sum = confusion_matrix.rowwise().sum();
        TMatrix normalized_confusion_matrix = confusion_matrix;
        for (int col=0; col < confusion_matrix.cols(); col++)
        {
            for (int row=0; row < confusion_matrix.rows(); row++)
            {
                normalized_confusion_matrix(row, col) /= row_sum(row);
            }
        }
        return normalized_confusion_matrix;
    }

};

}
