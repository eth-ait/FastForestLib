#include <memory>
#include <vector>

namespace AIT {

  /// @brief A decision forest, i.e. a collection of decision trees.
  template <typename split_point, typename statistics>
  class Forest {
    std::vector<std::unique_ptr<tree_type>> trees_;

  public:
    typedef Tree<split_point, statistics> tree_type;
    typedef tree_type::data_type data_type;
    typedef std::vector<int>::size_type size_type;

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
    std::vector<std::vector<size_type>> Evaluate(
      std::vector<data_type> data) const
    {
      std::vector<std::vector<size_type>> leaf_node_indices;
      Evaluate(data, leaf_node_indices);
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
    void Evaluate(
      std::vector<data_type> data,
      std::vector<std::vector<size_type>> &leaf_node_indices) const
    {
      leaf_node_indices.resize(NumOfTrees());
      for (size_type i=0; i < NumOfTrees(); i++) {
        //trees_[i]->resize(data.Count());
        //trees_[i]->Evaluate(data, leaf_node_indices[i]);
      }
    }

  };

}
