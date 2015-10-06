#pragma once

#include <vector>
#include <memory>
#include <iostream>

#ifdef SERIALIZE_WITH_BOOST
#include <boost/serialization/vector.hpp>
#else
#include <cereal/types/vector.hpp>
#endif


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

protected:
    struct NodeEntry
    {
        NodeType node;
        bool is_leaf;
        
        NodeEntry() : is_leaf(false)
        {}
        
#ifdef SERIALIZE_WITH_BOOST
        friend class boost::serialization::access;
#endif

        template <typename Archive>
        void serialize(Archive &archive, const unsigned int version)
        {
#ifdef SERIALIZE_WITH_BOOST
            archive & BOOST_SERIALIZATION_NVP(node);
            archive & BOOST_SERIALIZATION_NVP(is_leaf);
#else
            archive(cereal::make_nvp("node", node));
            archive(cereal::make_nvp("is_leaf", is_leaf));
#endif
        }
    };

    template <typename TreeType, typename ValueType>
    class NodeIterator_
    {
    protected:
        TreeType &tree_;
        size_type node_index_;

    public:
        NodeIterator_(TreeType &tree, size_type node_index)
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

        bool operator<(const NodeIterator_ &other) const
        {
            return this->node_index_ < other.node_index_;
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
        
        NodeIterator_ left_child()
        {
            assert(!this->is_leaf());
            size_type left_child_index = 2 * node_index_ + 1;
            return NodeIterator_(tree_, left_child_index);
        }
        
        NodeIterator_ right_child()
        {
            assert(!this->is_leaf());
            size_type right_child_index = 2 * node_index_ + 2;
            return NodeIterator_(tree_, right_child_index);
        }
        
        NodeIterator_ parent()
        {
            assert(!this->is_root_node());
            size_type parent_index = (node_index_ - 1) / 2;
            return NodeIterator_(tree_, parent_index);
        }
    };

    size_type depth_;
    std::vector<NodeEntry> node_entries_;

    // TODO: Use boost::iterator to implement a proper iterator
    class NodeEntryIteratorWrapper
    {
    public:
        using BaseIterator = typename std::vector<NodeEntry>::iterator;

    protected:
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

        bool operator<(const NodeEntryIteratorWrapper &other) const
        {
            return this->it_ < other.it_;
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
    using NodeIterator = NodeIterator_<Tree<TSplitPoint, TStatistics>, NodeType>;
    using ConstNodeIterator = NodeIterator_<Tree<TSplitPoint, TStatistics> const, NodeType const>;
    
    class TreeLevel
    {
    protected:
        Tree &tree_;
        size_type level_;

    public:
        using iterator = NodeEntryIteratorWrapper;
        using const_iterator = const NodeEntryIteratorWrapper;

        TreeLevel(Tree &tree, size_type level)
        : tree_(tree), level_(level)
        {}

        iterator begin() const
        {
            size_type offset = compute_node_offset(level_);
            return NodeEntryIteratorWrapper(tree_.node_entries_.begin() + offset);
        }

        iterator end() const
        {
            size_type offset = compute_node_offset(level_ + 1);
            return NodeEntryIteratorWrapper(tree_.node_entries_.begin() + offset);
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
    NodeIterator get_root()
    {
        return NodeIterator(*this, 0);
    }
    
    /// @brief Return a node in the tree.
    ConstNodeIterator get_root() const
    {
        return ConstNodeIterator(*this, 0);
    }

    /// @brief Return a node in the tree.
    NodeIterator get_node(size_type index)
    {
        return NodeIterator(*this, index);
    }
    
    /// @brief Return a node in the tree.
    ConstNodeIterator get_node(size_type index) const
    {
        return ConstNodeIterator(*this, index);
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
    void evaluate(const std::vector<TSample> &samples,
        std::vector<size_type> &leaf_node_indices) const
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
    void evaluate(const TSampleIterator &it_start, const TSampleIterator &it_end, std::vector<size_type> &leaf_node_indices) const
    {
        leaf_node_indices.reserve(it_end - it_start);
        for (auto it = it_start; it != it_end; it++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            leaf_node_indices.push_back(node_iter.get_index());
        }
    }

    /// @brief evaluate a collection of data-points on the tree and return the iterators (const version).
    /// @param data The collection of data-points
    /// @param max_depth The maximum depth to which the tree should be traversed
    template <typename TSample>
    const std::vector<ConstNodeIterator> evaluate(const std::vector<TSample> &samples, size_type max_depth = std::numeric_limits<size_type>::max()) const
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
    std::vector<NodeIterator> evaluate(const std::vector<TSample> &samples, size_type max_depth = std::numeric_limits<size_type>::max())
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

    template <typename TSample>
    std::vector<ConstNodeIterator> evaluate(const std::vector<TSample> &samples, size_type max_depth = std::numeric_limits<size_type>::max()) const
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
    

    /// @brief Evaluate a data-points on the tree (const version).
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename TSample>
    ConstNodeIterator evaluate(const TSample &sample, size_type max_depth = std::numeric_limits<size_type>::max()) const
    {
        size_type current_depth = 1;
        ConstNodeIterator node_iter = get_root();
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
    template <typename TSample>
    NodeIterator evaluate(const TSample &sample, size_type max_depth = std::numeric_limits<size_type>::max())
    {
        size_type current_depth = 1;
        NodeIterator node_iter = get_root();
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

    template <typename TSample>
    void evaluate_parallel(const std::vector<TSample> &samples, std::function<void(const TSample &, ConstNodeIterator &)> &func) const
    {
        //#pragma omp parallel for
        for (int i = 0; i < samples.size(); i++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(samples[i]);
            func(samples[i], node_iter);
        }
    }

    template <typename TSample>
    void evaluate_parallel(const std::vector<TSample> &samples, const std::function<void(const TSample &, const NodeType &)> &func) const
    {
        std::function<void(const TSample &, ConstNodeIterator &)> func_wrapper = [&func](const TSample &sample, ConstNodeIterator &node_iter)
        {
            func(sample, *node_iter);
        };
        evaluate_parallel(samples, func_wrapper);
    }

    template <typename TSample>
    void evaluate(const std::vector<TSample> &samples, std::function<void(const TSample &, ConstNodeIterator &)> &func) const
    {
        for (auto it = samples.cbegin(); it != samples.cend(); it++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            func(*it, node_iter);
        }
    }

    template <typename TSample>
    void evaluate(const std::vector<TSample> &samples, const std::function<void (const TSample &, const NodeType &)> &func) const
    {
        std::function<void (const TSample &, ConstNodeIterator &)> func_wrapper = [&func] (const TSample &sample, ConstNodeIterator &node_iter)
        {
            func(sample, *node_iter);
        };
        evaluate(samples, func_wrapper);
    }
    
    template <typename TSampleIterator>
    void evaluate(const TSampleIterator &it_start, const TSampleIterator &it_end, std::function<void(const TSampleIterator &, ConstNodeIterator &)> &func) const
    {
        for (auto it = it_start; it != it_end; it++)
        {
            ConstNodeIterator node_iter = evaluate_to_iterator(*it);
            func(it, node_iter);
        }
    }
    
    template <typename TSampleIterator>
    void evaluate(const TSampleIterator &it_start, const TSampleIterator &it_end, const std::function<void (const TSampleIterator &, const NodeType &)> &func) const
    {
        std::function<void (const TSampleIterator &, ConstNodeIterator &)> func_wrapper = [&func] (const TSampleIterator &sample_it, ConstNodeIterator &node_iter)
        {
            func(sample_it, *node_iter);
        };
        evaluate(it_start, it_end, func_wrapper);
    }

    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    template <typename TSample>
    ConstNodeIterator evaluate_to_iterator(const TSample &sample) const
    {
        ConstNodeIterator node_iter = get_root();
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
    
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;
#endif

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version)
    {
#ifdef SERIALIZE_WITH_BOOST
        archive & BOOST_SERIALIZATION_NVP(depth_);
        archive & BOOST_SERIALIZATION_NVP(node_entries_);
#else
        archive(cereal::make_nvp("depth", depth_));
        archive(cereal::make_nvp("node_entries", node_entries_));
#endif
    }

protected:
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

template <typename TSplitPoint, typename TStatistics, typename TSampleIterator, typename TMatrix = Eigen::MatrixXd>
class TreeUtilities
{
    using TreeType = Tree<TSplitPoint, TStatistics>;

    const TreeType &tree_;

public:
    TreeUtilities(const TreeType &tree)
    : tree_(tree)
    {}
    
//    // TODO: Samples should contain information on number of labels
//    template <int num_of_labels>
//    TMatrix compute_confusion_matrix(const std::vector<TSample> &samples) const
//    {
//        TMatrix confusion_matrix(num_of_labels, num_of_labels);
//        confusion_matrix.setZero();
//        /*tree_.template evaluate<TSample>(samples, [&confusion_matrix] (const TSample &sample, const typename TreeType::NodeType &node) {
//                            typename TSample::label_type true_label = sample.GetLabel();
//                            const TStatistics &statistics = node.GetStatistics();
//                            const auto &histogram = statistics.GetHistogram();
//                            typename TSample::label_type predicted_label = std::max_element(histogram.cbegin(), histogram.cend()) - histogram.cbegin();
//                            confusion_matrix(true_label, predicted_label)++;
//        });*/
//        tree_.template evaluate_parallel<TSample>(samples, [&confusion_matrix](const TSample &sample, const typename TreeType::NodeType &node)
//        {
//            size_type true_label = sample.get_label();
//            const TStatistics &statistics = node.get_statistics();
//            const auto &histogram = statistics.get_histogram();
//            size_type predicted_label = std::max_element(histogram.cbegin(), histogram.cend()) - histogram.cbegin();
//            confusion_matrix(true_label, predicted_label)++;
//        });
//        return confusion_matrix;
//    }

    // TODO: Samples should contain information on number of labels
    template <int num_of_labels>
    TMatrix compute_confusion_matrix(const TSampleIterator &it_start, const TSampleIterator &it_end) const
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
        tree_.template evaluate<TSampleIterator>(it_start, it_end, [&confusion_matrix](const TSampleIterator &it, const typename TreeType::NodeType &node)
                                                  {
                                                      size_type true_label = it->get_label();
                                                      const TStatistics &statistics = node.get_statistics();
                                                      const auto &histogram = statistics.get_histogram();
                                                      size_type predicted_label = std::max_element(histogram.cbegin(), histogram.cend()) - histogram.cbegin();
                                                      confusion_matrix(true_label, predicted_label)++;
                                                  });
        return confusion_matrix;
    }
    
    template <int num_of_labels>
    TMatrix compute_normalized_confusion_matrix(const TSampleIterator &it_start, const TSampleIterator &it_end) const
    {
        TMatrix confusion_matrix = compute_confusion_matrix<num_of_labels>(it_start, it_end);
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
