#ifndef AITDistributedRandomForest_node_h
#define AITDistributedRandomForest_node_h

#include <memory>
#include <vector>

#include "data_point_collection.h"

namespace AIT {
    
  /// @brief A node of a decision tree.
  template <typename SplitPoint, typename Statistics>
  class Node {
  public:
    enum class Direction {LEFT=-1, RIGHT=+1};

    typedef std::vector<int>::size_type size_type;

    virtual Direction Evaluate(const DataPointCollection<> &data_point_collection, size_type index) const = 0;

  };

}

#endif
