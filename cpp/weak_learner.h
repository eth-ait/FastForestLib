#ifndef AITDistributedRandomForest_weak_learner_h
#define AITDistributedRandomForest_weak_learner_h

#include <tuple>
#include <iostream>

namespace AIT {

  template <typename TSplitPoint, typename TStatistics, typename TIterator, typename value_type=double, typename size_type=std::size_t>
  class WeakLearner {
  public:
      virtual TStatistics ComputeStatistics(TIterator first_sample, TIterator last_sample) const = 0;
      virtual std::vector<TSplitPoint> SampleSplitPoints(TIterator first_sample, TIterator last_sample, size_type num_of_features, size_type num_of_thresholds) const = 0;
      virtual std::vector<TStatistics> ComputeSplitStatistics(TIterator first_sample, TIterator last_sample, const std::vector<TSplitPoint> &split_points) const = 0;
      virtual std::tuple<size_type, value_type> FindBestSplitPointIndex(const TStatistics &current_statistics, const std::vector<TStatistics> &split_statistics) const = 0;
      virtual size_type Partition(TIterator first_sample, TIterator last_sample, const TSplitPoint &best_split_point) const = 0;
  };

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
//  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}

#endif
