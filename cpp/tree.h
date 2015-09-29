#pragma once

#include <vector>
#include <memory>
#include <iostream>

#include <cereal/types/vector.hpp>

#include "node.h"

// TODO: Modify iterator interface to reflect STL usage.
namespace ait {

/// @brief A decision tree.
template <typename TSplitPoint, typename TStatistics>
class Tree {
public:
    typedef Node<TSplitPoint, TStatistics> NodeType;
    typedef std::vector<int>::size_type size_type;

private:
    size_type depth_;
    size_type first_leaf_node_index_;
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
        
        size_type get_node_index() const
        {
            return node_index_;
        }
        
        bool is_root_node() const
        {
            return node_index_ == 0;
        }
        
        bool is_leaf_node() const
        {
            return node_index_ >= tree_ptr_->first_leaf_node_index_;
        }
        
        void goto_left_child()
        {
            assert(!this->is_leaf_node());
            size_type left_child_index = 2 * node_index_ + 1;
            node_index_ = left_child_index;
        }
        
        void goto_right_child()
        {
            assert(!this->is_leaf_node());
            size_type right_child_index = 2 * node_index_ + 2;
            node_index_ = right_child_index;
        }
        
        void goto_parent()
        {
            assert(!this->is_root_node());
            size_type parent_index = (node_index_ - 1) / 2;
            node_index_ = parent_index;
        }
        
        NodeIterator_ left_child()
        {
            assert(!this->is_leaf_node());
            size_type left_child_index = 2 * node_index_ + 1;
            return NodeIterator_(tree_ptr_, left_child_index);
        }
        
        NodeIterator_ right_child()
        {
            assert(!this->is_leaf_node());
            size_type right_child_index = 2 * node_index_ + 2;
            return NodeIterator_(tree_ptr_, right_child_index);
        }
        
        NodeIterator_ parent()
        {
            assert(!this->is_root_node());
            size_type parent_index = (node_index_ - 1) / 2;
            return NodeIterator_(tree_ptr_, parent_index);
        }
    };

public:
    typedef NodeIterator_<Tree<TSplitPoint, TStatistics>, NodeType> NodeIterator;
    typedef NodeIterator_<Tree<TSplitPoint, TStatistics> const, NodeType const> ConstNodeIterator;
    
    /// @brief Create an empty tree with no nodes.
    Tree()
    : depth_(0), first_leaf_node_index_(0)
    {}

    /// @brief Create a tree.
    /// @param depth The depth of the tree. A depth of 1 corresponds to a tree
    ///              with a single node.
    Tree(size_type depth)
    : depth_(depth), first_leaf_node_index_((1 << (depth - 1)) - 1)
    {
        size_type num_of_nodes = (1 << depth_) - 1;
        nodes_.resize(num_of_nodes);
    }

    /// @brief Return a node in the tree.
    NodeIterator get_root()
    {
        return NodeIterator(this, 0);
    }
    
    /// @brief Return a node in the tree.
    ConstNodeIterator get_root() const
    {
        return ConstNodeIterator(this, 0);
    }

    /// @brief Return a node in the tree.
    NodeIterator get_node(size_type index)
    {
        return NodeIterator(this, index);
    }
    
    /// @brief Return a node in the tree.
    ConstNodeIterator get_node(size_type index) const
    {
        return ConstNodeIterator(this, index);
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
        return nodes_.size();
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
        for (auto it = samples.cbegin(); it != samples.cend(); it++) {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            leaf_node_indices.push_back(node_iter.get_node_index());
        }
    }
    
    template <typename Sample>
    const std::vector<ConstNodeIterator> evaluate(const std::vector<Sample> &samples) const
    {
        std::vector<ConstNodeIterator> leaf_nodes;
        leaf_nodes.reserve(samples.size());
        for (auto it = samples.cbegin(); it != samples.cend(); it++) {
            ConstNodeIterator node_iter = evaluate(*it);
            leaf_nodes.push_back(node_iter);
        }
        return leaf_nodes;
    }
    
    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename Sample>
    size_type evaluate(const Sample &sample) const
    {
        ConstNodeIterator node_iter = get_root();
        while (!node_iter.IsLeafNode()) {
            Direction direction = node_iter->get_split_point().evaluate(sample);
            if (direction == Direction::LEFT)
                node_iter.goto_left_child();
            else
                node_iter.goto_right_child();
        }
        return node_iter.get_node_index();
    }

    template <typename Sample>
    void evaluate_parallel(const std::vector<Sample> &samples, std::function<void(const Sample &, ConstNodeIterator &)> &func) const
    {
        //#pragma omp parallel for
        for (int i = 0; i < samples.size(); i++) {
            ConstNodeIterator node_iter = evaluate_to_iterator(samples[i]);
            func(samples[i], node_iter);
        }
    }

    template <typename Sample>
    void evaluate_parallel(const std::vector<Sample> &samples, const std::function<void(const Sample &, const NodeType &)> &func) const
    {
        std::function<void(const Sample &, ConstNodeIterator &)> func_wrapper = [&func](const Sample &sample, ConstNodeIterator &node_iter) {
            func(sample, *node_iter);
        };
        evaluate_parallel(samples, func_wrapper);
    }

    template <typename Sample>
    void evaluate(const std::vector<Sample> &samples, std::function<void(const Sample &, ConstNodeIterator &)> &func) const
    {
        for (auto it = samples.cbegin(); it != samples.cend(); it++) {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            func(*it, node_iter);
        }
    }

    template <typename Sample>
    void evaluate(const std::vector<Sample> &samples, const std::function<void (const Sample &, const NodeType &)> &func) const
    {
        std::function<void (const Sample &, ConstNodeIterator &)> func_wrapper = [&func] (const Sample &sample, ConstNodeIterator &node_iter) {
            func(sample, *node_iter);
        };
        evaluate(samples, func_wrapper);
    }

    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename Sample>
    ConstNodeIterator evaluate_to_iterator(const Sample &sample) const
    {
        ConstNodeIterator node_iter = get_root();
        while (!node_iter.is_leaf_node()) {
            Direction direction = node_iter->get_split_point().evaluate(sample);
            if (direction == Direction::LEFT)
                node_iter.goto_left_child();
            else
                node_iter.goto_right_child();
        }
        return node_iter;
    }

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version)
    {
        archive(cereal::make_nvp("depth", depth_));
        archive(cereal::make_nvp("first_leaf_node_index", first_leaf_node_index_));
//            archive(cereal::make_nvp("nodes", nodes_));
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

    // @brief Check if the specified node is a leaf node.
    bool is_leaf_node_index(size_type index) const
    {
        if (index >= first_leaf_node_index_)
            return true;
        else
            return false;
    }

};

template <typename TSplitPoint, typename TStatistics, typename TSample, typename TMatrix = Eigen::MatrixXd>
class TreeUtilities {
    typedef Tree<TSplitPoint, TStatistics> TreeType;
    
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
        tree_.template evaluate_parallel<TSample>(samples, [&confusion_matrix](const TSample &sample, const typename TreeType::NodeType &node) {
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
        for (int col=0; col < confusion_matrix.cols(); col++) {
            for (int row=0; row < confusion_matrix.rows(); row++) {
                normalized_confusion_matrix(row, col) /= row_sum(row);
            }
        }
        return normalized_confusion_matrix;
    }

};

}
