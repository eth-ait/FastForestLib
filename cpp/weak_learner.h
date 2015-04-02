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
        typedef typename std::vector<TStatistics>::size_type size_type;

		SplitStatistics(std::vector<TStatistics> &&left_statistics_collection, std::vector<TStatistics> &&right_statistics_collection)
			: left_statistics_collection_(left_statistics_collection), right_statistics_collection_(right_statistics_collection)
		{
            if (left_statistics_collection_.size() != right_statistics_collection_.size())
                throw std::runtime_error("The vectors of left and right child statistics must have the same size");
        }

        size_type Count() const {
            return left_statistics_collection_.size();
        }

        std::tuple<const TStatistics &, const TStatistics &> GetSplitStatistics(size_type index) const {
            return std::make_tuple<const TStatistics &, const TStatistics &>(left_statistics_collection_[index], right_statistics_collection_[index]);
        }
        
        const TStatistics & GetLeftStatistics(size_type index) const {
            return left_statistics_collection_[index];
        }
        
        const TStatistics & GetRightStatistics(size_type index) const {
            return right_statistics_collection_[index];
        }

		const std::vector<TStatistics> & LeftStatisticsCollection() const {
			return left_statistics_collection_;
		}

		const std::vector<TStatistics> & RightStatisticsCollection() const {
			return right_statistics_collection_;
		}

	};

    template <typename TSplitPoint, typename TStatistics, typename TIterator, typename TRandomEngine, typename t_size_type = std::size_t>
    class WeakLearner {
	public:
		typedef TStatistics Statistics;
		typedef TSplitPoint SplitPoint;
		typedef TIterator Iterator;
		typedef t_size_type size_type;
		typedef typename TStatistics::entropy_type entropy_type;

        WeakLearner()
        {}
        
		virtual ~WeakLearner()
        {}

		virtual TStatistics ComputeStatistics(TIterator first_sample, TIterator last_sample) const = 0;

        virtual std::vector<TSplitPoint> SampleSplitPoints(TIterator first_sample, TIterator last_sample, size_type num_of_features, size_type num_of_thresholds, TRandomEngine &rnd_engine) const = 0;

        virtual SplitStatistics<TStatistics> ComputeSplitStatistics(TIterator first_sample, TIterator last_sample, const std::vector<TSplitPoint> &split_points) const = 0;

        virtual std::tuple<size_type, typename TStatistics::entropy_type> FindBestSplitPoint(const TStatistics &current_statistics, const SplitStatistics<TStatistics> &split_statistics) const = 0;

        virtual TIterator Partition(TIterator first_sample, TIterator last_sample, const TSplitPoint &split_point) const {
            TIterator it_left = first_sample;
            TIterator it_right = last_sample - 1;
            while (it_left < it_right) {
                Direction direction = split_point.Evaluate(*it_left);
                if (direction == Direction::LEFT)
                    it_left++;
                else {
                    std::swap(*it_left, *it_right);
                    it_right--;
                }
            }
            
            Direction direction = split_point.Evaluate(*it_left);
            TIterator it_split;
            if (direction == Direction::LEFT)
                it_split = it_left + 1;
            else
                it_split = it_left;
            
            // check partitioning
            for (TIterator it = first_sample; it != it_split; it++) {
                Direction dir = split_point.Evaluate(*it);
                if (dir != Direction::LEFT)
                    throw std::runtime_error("Samples are not partitioned properly.");
            }
            for (TIterator it = it_split; it != last_sample; it++) {
                Direction dir = split_point.Evaluate(*it);
                if (dir != Direction::RIGHT)
                    throw std::runtime_error("Samples are not partitioned properly.");
            }
            
            return it_split;
        }

	};

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
//  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}

#endif
