#pragma once

#include <memory>
#include <vector>
#include <functional>

#include <cereal/types/vector.hpp>
#include <Eigen/Dense>

#include "ait.h"
#include "tree.h"

namespace ait
{

/// @brief A decision forest, i.e. a collection of decision trees.
template <typename TSplitPoint, typename TStatistics>
class Forest
{
public:
    using TreeType = Tree<TSplitPoint, TStatistics>;
    using NodeType = typename TreeType::NodeType;

private:
    std::vector<TreeType> trees_;

public:

    /// @brief Create an empty forest.
    Forest() {}

    /// @brief Add a tree to the forest.
    void add_tree(const TreeType &tree)
    {
        trees_.push_back(tree);
    }
    
    /// @brief Add a tree to the forest.
    void add_tree(TreeType &&tree)
    {
        trees_.push_back(std::move(tree));
    }
    
    /// @brief Return a tree in the forest.
    const TreeType & get_tree(size_type index) const
    {
        return trees_[index];
    }

    /// @brief Return a tree in the forest.
    TreeType & get_tree(size_type index)
    {
        return trees_[index];
    }

    /// @brief Return number of trees in the forest.
    size_type size() const
    {
        return trees_.size();
    }

    // TODO: Think about evaluation methods
    template <typename Sample>
    void evaluate(const std::vector<Sample> &samples, const std::function<void (const Sample &sample, const typename TreeType::ConstTreeIterator &)> &func) const {
        for (size_type i=0; i < size(); i++) {
            const TreeType &tree = trees_[i];
            for (auto it = samples.cbegin(); it != samples.cend(); it++) {
                typename TreeType::ConstTreeIterator node_iter = tree.evaluate_to_iterator(*it);
                func(*it, node_iter);
            }
        }
    }

    template <typename Sample>
    void evaluate(const std::vector<Sample> &samples, const std::function<void (const Sample &sample, const NodeType &)> &func) const {
        evaluate(samples, [&func] (const Sample &sample, const typename TreeType::ConstTreeIterator &node_iter) {
//            func(sample, *node_iter);
        });
    }

    /// @brief Evaluate a collection of data-points on each tree in the forest.
    /// @param data The collection of data-points
    /// @return A vector of the results. See #Evaluate().
    template <typename Sample>
    std::vector<std::vector<size_type> > evaluate(const std::vector<Sample> &samples) const
    {
        std::vector<std::vector<size_type> > forest_leaf_node_indices;
        evaluate(samples, forest_leaf_node_indices);
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
    void evaluate(const std::vector<Sample> &samples, std::vector<std::vector<size_type> > &forest_leaf_node_indices) const
    {
        forest_leaf_node_indices.reserve(size());
        for (size_type i=0; i < size(); i++) {
            std::vector<size_type> leaf_node_indices;
            leaf_node_indices.reserve(samples.size());
            trees_[i].evaluate(samples, leaf_node_indices);
            forest_leaf_node_indices.push_back(std::move(leaf_node_indices));
        }
    }

    template <typename Sample>
    const std::vector<std::vector<typename TreeType::ConstTreeIterator> > evaluate_to_iterator(const std::vector<Sample> &samples) const
    {
        std::vector<std::vector<typename TreeType::ConstTreeIterator> > forest_leaf_nodes;
        forest_leaf_nodes.reserve(size());
        for (size_type i=0; i < size(); i++) {
            std::vector<typename TreeType::ConstTreeIterator> leaf_nodes;
            leaf_nodes.reserve(samples.size());
            for (auto it = samples.cbegin(); it != samples.cend(); it++) {
                typename TreeType::ConstTreeIterator node_iter = Evaluate(*it);
                leaf_nodes.push_back(node_iter);
            }
            forest_leaf_nodes.push_back(std::move(leaf_nodes));
        }
        return forest_leaf_nodes;
    }

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version)
    {
        archive(cereal::make_nvp("trees", trees_));
    }

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
