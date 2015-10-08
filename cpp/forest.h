//
//  forest.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

#include <memory>
#include <vector>
#include <functional>

#include <boost/utility/enable_if.hpp>
#include <boost/mpl/has_xxx.hpp>
#ifdef SERIALIZE_WITH_BOOST
#include <boost/serialization/vector.hpp>
#endif
#include <cereal/types/vector.hpp>
#include <Eigen/Dense>

#include "ait.h"
#include "tree.h"
#include "mpl_utils.h"

namespace ait
{

/// @brief A decision forest, i.e. a collection of decision trees.
template <typename TSplitPoint, typename TStatistics>
class Forest
{
public:
    using TreeT = Tree<TSplitPoint, TStatistics>;
    using NodeT = typename TreeT::NodeT;

public:
    using TreeIterator = typename std::vector<TreeT>::iterator;
    using ConstTreeIterator = typename std::vector<TreeT>::const_iterator;
    using iterator = TreeIterator;
    using const_iterator = ConstTreeIterator;

    /// @brief Create an empty forest.
    Forest() {}

    /// @brief Add a tree to the forest.
    void add_tree(const TreeT &tree)
    {
        trees_.push_back(tree);
    }
    
    /// @brief Add a tree to the forest.
    void add_tree(TreeT &&tree)
    {
        trees_.push_back(std::move(tree));
    }
    
    TreeIterator begin()
    {
        return trees_.begin();
    }
    
    TreeIterator end()
    {
        return trees_.end();
    }

    ConstTreeIterator cbegin() const
    {
        return trees_.cbegin();
    }
    
    ConstTreeIterator cend() const
    {
        return trees_.cend();
    }

    /// @brief Return a tree in the forest.
    const TreeT & get_tree(size_type index) const
    {
        return trees_[index];
    }

    /// @brief Return a tree in the forest.
    TreeT & get_tree(size_type index)
    {
        return trees_[index];
    }

    /// @brief Return number of trees in the forest.
    size_type size() const
    {
        return trees_.size();
    }

    // TODO: Think about evaluation methods
    template <typename TSample>
    void evaluate(const std::vector<TSample> &samples, const std::function<void (const TSample &sample, const typename TreeT::ConstNodeIterator &)> &func) const {
        for (size_type i=0; i < size(); i++) {
            const TreeT &tree = trees_[i];
            for (auto it = samples.cbegin(); it != samples.cend(); it++)
            {
                typename TreeT::ConstNodeIterator node_iter = tree.evaluate_to_iterator(*it);
                func(*it, node_iter);
            }
        }
    }
    
    template <typename TSampleIterator>
    void evaluate(const TSampleIterator &it_start, const TSampleIterator &it_end, const std::function<void (const TSampleIterator &, const typename TreeT::ConstNodeIterator &)> &func) const {
        for (size_type i=0; i < size(); i++) {
            const TreeT &tree = trees_[i];
            for (auto it = it_start; it != it_end; ++it)
            {
                typename TreeT::ConstNodeIterator node_iter = tree.evaluate_to_iterator(*it);
                func(*it, node_iter);
            }
        }
    }

    template <typename TSample>
    void evaluate(const std::vector<TSample> &samples, const std::function<void (const TSample &sample, const NodeT &)> &func) const {
        evaluate(samples, [&func] (const TSample &sample, const typename TreeT::ConstNodeIterator &node_iter)
        {
//            func(sample, *node_iter);
        });
    }

    /// @brief Evaluate a collection of samples on each tree in the forest.
    /// @param samples The collection of samples.
    /// @return A vector of the results. See #Evaluate().
    template <typename TSample>
    std::vector<std::vector<size_type>> evaluate(const std::vector<TSample> &samples) const
    {
        std::vector<std::vector<size_type>> forest_leaf_node_indices;
        evaluate(samples, forest_leaf_node_indices);
        return forest_leaf_node_indices;
    }
    
    /// @brief Evaluate a collection of data-points on each tree in the forest.
    /// @param it_start The first sample iterator.
    /// @param it_end The last sample iterator.
    /// @return A vector of the results. See #Evaluate().
    template <typename TSampleIterator>
    std::vector<std::vector<size_type>> evaluate(const TSampleIterator &it_start, const TSampleIterator &it_end) const
    {
        std::vector<std::vector<size_type>> forest_leaf_node_indices;
        evaluate(it_start, it_end, forest_leaf_node_indices);
        return forest_leaf_node_indices;
    }

    /// @brief Evaluate a collection of samples on each tree in the forest.
    /// @param samples The collection of samples
    /// @param forest_leaf_node_indices A vector for storing the results. For each tree
    ///                          it will contain another vector storing the
    ///                          index of the corresponding leaf node for each
    ///                          data-point.
    ///                          So leaf_node_indices.size() will be equal to
    ///                          NumOfTrees().
    template <typename TSample>
    void evaluate(const std::vector<TSample> &samples, std::vector<std::vector<size_type>> &forest_leaf_node_indices) const
    {
        forest_leaf_node_indices.reserve(size());
        for (size_type i=0; i < size(); i++)
        {
            std::vector<size_type> leaf_node_indices;
            leaf_node_indices.reserve(samples.size());
            trees_[i].evaluate(samples, leaf_node_indices);
            forest_leaf_node_indices.push_back(std::move(leaf_node_indices));
        }
    }
    
    /// @brief Evaluate a collection of samples on each tree in the forest.
    /// @param it_start The first sample iterator.
    /// @param it_end The last sample iterator.
    /// @param forest_leaf_node_indices A vector for storing the results. For each tree
    ///                          it will contain another vector storing the
    ///                          index of the corresponding leaf node for each
    ///                          data-point.
    ///                          So leaf_node_indices.size() will be equal to
    ///                          NumOfTrees().
    template <typename TSampleIterator>
    void evaluate(const TSampleIterator &it_start, const TSampleIterator &it_end, std::vector<std::vector<size_type>> &forest_leaf_node_indices) const
    {
        forest_leaf_node_indices.reserve(size());
        for (size_type i=0; i < size(); i++)
        {
            std::vector<size_type> leaf_node_indices;
            leaf_node_indices.reserve(it_end - it_start);
            trees_[i].evaluate(it_start, it_end, leaf_node_indices);
            forest_leaf_node_indices.push_back(std::move(leaf_node_indices));
        }
    }

    template <typename TSample>
    const std::vector<std::vector<typename TreeT::ConstNodeIterator>> evaluate_to_iterator(const std::vector<TSample> &samples) const
    {
        std::vector<std::vector<typename TreeT::ConstNodeIterator>> forest_leaf_nodes;
        forest_leaf_nodes.reserve(size());
        for (size_type i=0; i < size(); i++)
        {
            std::vector<typename TreeT::ConstNodeIterator> leaf_nodes;
            leaf_nodes.reserve(samples.size());
            for (auto it = samples.cbegin(); it != samples.cend(); it++)
            {
                typename TreeT::ConstNodeIterator node_iter = Evaluate(*it);
                leaf_nodes.push_back(node_iter);
            }
            forest_leaf_nodes.push_back(std::move(leaf_nodes));
        }
        return forest_leaf_nodes;
    }

private:
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive & trees_;
    }
#endif
    
    friend class cereal::access;

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("trees", trees_));
    }

    std::vector<TreeT> trees_;

};

template <typename TSplitPoint, typename TStatistics, typename TMatrix = Eigen::MatrixXd>
class ForestUtilities
{
    using ForestType = Forest<TSplitPoint, TStatistics>;
    
    const ForestType &forest_;
public:
    ForestUtilities(const ForestType &forest)
    : forest_(forest)
    {}

//    MatrixType compute_confusion_matrix() const {
//    }
};

}
