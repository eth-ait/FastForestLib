#include <memory>
#include <vector>

namespace AIT {

  /// @brief A decision tree.
  template <typename split_point, typename statistics>
  class Tree {
    const size_type depth_;
    const size_type first_leaf_node_index_;
    std::vector<Node> nodes_;

  public:
    typedef Node<split_point, statistics> node_type;
    typedef std::vector<int>::size_type size_type;
    typedef node_type::data_type data_type;

    /// @brief Create a tree.
    /// @param depth The depth of the tree. A depth of 0 corresponds to a tree
    ///              with a single node.
    Tree(size_type depth) : depth_(depth),
                            first_leaf_node_index_((1 << depth) - 1) {
      size_type num_of_nodes = (1 << (depth_ + 1)) - 1;
      nodes_.resize(depth_);
    }

    /// @brief Return a node in the tree.
    const node_type& GetNode(size_type index) const
    {
      return nodes_[index];
    }

    /// @brief Return a node in the tree.
    node_type& GetTree(size_type index)
    {
      return nodes_[index];
    }

    /// @brief Return depth of the tree. A depth of 0 corresponds to a tree
    ///        with a single node.
    size_type Depth() const {
      return depth_;
    }

    /// @brief Return number of nodes in the tree.
    size_type NumOfNodes() const
    {
      return nodes_.size();
    }

    /// @brief Evaluate a collection of data-points on the tree.
    /// @param data The collection of data-points
    /// @return A vector of the results. See #Evaluate().
    std::vector<size_type> Evaluate(std::vector<double> data) const
    {
      std::vector<size_type> leaf_node_indices;
      Evaluate(data, leaf_node_indices);
      return leaf_node_indices;
    }

    /// @brief Evaluate a collection of data-points on the tree.
    /// @param data The collection of data-points
    /// @param leaf_node_indices A vector for storing the results. For each
    ///                          data-point it will contain the index of the
    ///                          corresponding leaf node.
    void Evaluate(
      std::vector<double> data,
      std::vector<size_type> &leaf_node_indices) const
    {
      leaf_node_indices.resize(data.size());
      for (size_type i=0; i < data.size(); i++) {
        leaf_node_indices[i] = Evaluate(data[i]);
      }
    }

    /// @brief Evaluate a data-points on the tree.
    /// @param data_point The data-point.
    /// @return The index of the corresponding leaf-node.
    size_type Evaluate(const &data_type data_point) const {
      size_type current_index = 0;
      while (!isLeafNodeIndex(current_index)) {
        Node::Direction direction = nodes_[current_index].Evaluate(data_point);
        if (direction == Node::Direction::LEFT)
          current_index = GetLeftChildIndex(current_index);
        else
          current_index = GetRightChildIndex(current_index);
      }
      return current_index;
    }

  private:
    // @brief Return the left child index of the specified node.
    size_type GetLeftChildIndex(size_type index) const {
      return 2 * index + 1;
    }

    // @brief Return the right child index of the specified node.
    size_type GetRightChildIndex(size_type index) const {
      return 2 * index + 2;
    }

    // @brief Check if the specified node is a leaf node.
    bool IsLeafNodeIndex(size_type index) const {
      if (index >= first_leaf_node_index_)
        return true;
      else
        return false;
    }

  };

}
