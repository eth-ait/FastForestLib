#include <tuple>
#include <iostream>

#include "weak_learner.h"
#include "histogram_statistics.h"

namespace AIT {

  class ImageSplitPoint {
  };

  template <typename size_type=std::size_t>
  class ImageSplitPointContext : SplitPointContext<ImageSplitPoint, size_type> {
    std::vector<ImageSplitPoint> split_points_;
  public:
      split_point getSplitPoint(size_type split_point_id) {
        return split_points_[split_point_id];
      }
  };

  template <typename value_type=double, typename size_type=std::size_t>
  class ImageWeakLearner : WeakLearner<ImageSplitPointContext, HistogramStatistics, value_type, size_type> {
  public:
      virtual HistogramStatistics computeStatistics(RandomAccessIterator first_sample, RandomAccessIterator last_sample) = 0;
      virtual ImageSplitPointContext sampleSplitPoints(RandomAccessIterator first_sample, RandomAccessIterator last_sample, size_type num_of_features, size_type num_of_thresholds) = 0;
      virtual SplitStatistics computeSplitStatistics(RandomAccessIterator first_sample, RandomAccessIterator last_sample, const split_point_context &split_point_context) = 0;
      virtual std::tuple<size_type, value_type> select_best_split_point(const statistics &current_statistics, const SplitStatistics &split_statistics) = 0;
      virtual size_type partition(RandomAccessIterator first_sample, RandomAccessIterator last_sample, SplitPointContext::SplitPoint best_split_point) = 0;
  };

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}
