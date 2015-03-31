#ifndef AITDistributedRandomForest_image_weak_learner_h
#define AITDistributedRandomForest_image_weak_learner_h

#include <tuple>
#include <iostream>

#include "weak_learner.h"
#include "histogram_statistics.h"

namespace AIT {

  class ImageSplitPoint {
  };

  template <typename TIterator, typename TStatistics, typename value_type=double, typename size_type=std::size_t>
  class ImageWeakLearner : WeakLearner<ImageSplitPoint, TStatistics, value_type, size_type> {
  public:
      typedef double entropy_type;

      virtual TStatistics ComputeStatistics(TIterator first_sample, TIterator last_sample) const {
          TStatistics statistics;
          for (TIterator it=first_sample; it != last_sample; it++) {
              statistics.Accumulate(*it);
          }
          return statistics;
      }

      virtual std::vector<ImageSplitPoint> SampleSplitPoints(TIterator first_sample, TIterator last_sample, size_type num_of_features, size_type num_of_thresholds) const {
          std::vector<ImageSplitPoint> split_points;
          // TODO
          for (size_type i_f=0; i_f < num_of_features; i_f++) {
              for (size_type i_t=0; i_t < num_of_thresholds; i_t++) {
                  split_points.push_back(ImageSplitPoint());
              }
          }
          return split_points;
      }

      virtual std::vector<TStatistics> ComputeSplitStatistics(TIterator first_sample, TIterator last_sample, const std::vector<ImageSplitPoint> &split_points) const {
          std::vector<TStatistics> split_statistics;
          for (auto it=split_points.cbegin(); it != split_points.cend(); it++) {
              TStatistics statistics;
              for (TIterator it=first_sample; it != last_sample; it++) {
                  statistics.Accumulate(*it);
              }
              split_statistics.push_back(statistics);
          }
          return split_statistics;
      }

      virtual std::tuple<size_type, value_type> FindBestSplitPointIndex(const TStatistics &current_statistics, const std::vector<TStatistics> &split_statistics) const {
          // TODO
          size_type best_split_point_index = 0;
          value_type best_split_point_entropy = split_statistics[0].Entropy();
//          for (size_type i=1; i < )
//          std::tuple<size_type, value_type> best_split_point_tuple;
          return std::make_tuple(best_split_point_index, best_split_point_entropy);
      }

      virtual size_type Partition(TIterator first_sample, TIterator last_sample, const ImageSplitPoint &best_split_point) const {
          return 0;
      }
  };

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
//  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}

#endif
