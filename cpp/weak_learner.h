#ifndef AITDistributedRandomForest_weak_learner_h
#define AITDistributedRandomForest_weak_learner_h

#include <tuple>
#include <iostream>

namespace AIT {

	template <typename TStatistics>
	class SplitStatistics {
		std::vector<TStatistics> left_statistics_collection_;
		std::vector<TStatistics> right_statistics_collection_;

	public:
		SplitStatistics(std::vector<TStatistics> &&left_statistics_collection, std::vector<TStatistics> &&right_statistics_collection)
			: left_statistics_collection_(left_statistics_collection), right_statistics_collection_(right_statistics_collection)
		{}

		const std::vector<TStatistics> & LeftStatisticsCollection() const {
			return left_statistics_collection_;
		}

		const std::vector<TStatistics> & RightStatisticsCollection() const {
			return right_statistics_collection_;
		}

	};

	template <typename TSplitPoint, typename TStatistics, typename TIterator, typename t_size_type=std::size_t>
	class WeakLearner {
	public:
		typedef TStatistics Statistics;
		typedef TSplitPoint SplitPoint;
		typedef TIterator Iterator;
		typedef t_size_type size_type;
		typedef typename TStatistics::entropy_type entropy_type;

		virtual TStatistics ComputeStatistics(TIterator first_sample, TIterator last_sample) const = 0;
		virtual std::vector<TSplitPoint> SampleSplitPoints(TIterator first_sample, TIterator last_sample, size_type num_of_features, size_type num_of_thresholds) const = 0;
		virtual SplitStatistics<TStatistics> ComputeSplitStatistics(TIterator first_sample, TIterator last_sample, const std::vector<TSplitPoint> &split_points) const = 0;
		virtual std::tuple<size_type, typename TStatistics::entropy_type> FindBestSplitPointIndex(const TStatistics &current_statistics, const std::vector<TStatistics> &split_statistics) const = 0;
		virtual TIterator Partition(TIterator it_left, TIterator it_right, const TSplitPoint &best_split_point) const = 0;
	};

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
//  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}

#endif
