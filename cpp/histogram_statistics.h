#ifndef AITDistributedRandomForest_statistics_h
#define AITDistributedRandomForest_statistics_h

#include <vector>
#include <cmath>
#include <numeric>

namespace AIT {

  /// @brief A histogram over the classes of samples.
  template <typename value_type=int,
            typename entropy_type=double,
            typename size_type=std::size_t>
  class HistogramStatistics {
    std::vector<value_type> histogram_;
    size_type num_of_samples_;

  public:
    // TODO: unused
    /// @brief Create an empty histogram.
    HistogramStatistics()
    : num_of_samples_(0) {}
      
    // TODO: unused
    /// @brief Create an empty histogram.
    /// @param num_of_classes The number of classes.
    HistogramStatistics(size_type num_of_classes)
    : histogram_(num_of_classes, 0),
      num_of_samples_(0) {}

    /// @brief Create a histogram from a vector of counts per class.
    HistogramStatistics(const std::vector<value_type> &histogram)
    : histogram_(histogram),
      num_of_samples_(std::accumulate(histogram.cbegin(), histogram.cend())) {}

    /// @brief Return the numbers of samples contributing to the histogram.
    size_type NumOfSamples() const {
      return num_of_samples_;
    }

    /// @brief Return the vector of counts per class.
    const std::vector<value_type> & Histogram() const {
      return histogram_;
    }

    /// @return: The Shannon entropy of the histogram.
    const entropy_type Entropy() const {
      entropy_type entropy = 0;
      for (auto it=histogram_.cbegin(); it != histogram_.cend(); it++) {
        const entropy_type count = static_cast<entropy_type>(*it);
        if (count > 0) {
          const entropy_type relative_count = count / num_of_samples_;
          entropy -= relative_count * std::log2(relative_count);
        }
      }
      return entropy;
    }

  };

}

#endif
