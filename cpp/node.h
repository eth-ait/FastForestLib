#include <memory>
#include <vector>

namespace AIT {

  /// @brief A node of a decision tree.
  template <typename SplitPoint, typename Statistics, typename data_type>
  class Node {
  public:
    enum class Direction {LEFT, RIGHT};

    typedef std::vector<int>::size_type size_type;
    typedef typename data_type data_type;

    virtual Direction Evaluate(const data_type &data_point) const = 0;

  };

}
