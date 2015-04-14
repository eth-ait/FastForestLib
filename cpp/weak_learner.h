#ifndef AITDistributedRandomForest_weak_learner_h
#define AITDistributedRandomForest_weak_learner_h

#include <tuple>
#include <iostream>

namespace AIT {
    
    template <typename TFeature, typename TThreshold>
    class SplitPoint {
        TFeature feature_;
        TThreshold threshold_;

    public:
        SplitPoint()
        {}

        SplitPoint(const TFeature &feature, const TThreshold &threshold)
        : feature_(feature), threshold_(threshold)
        {}

        template <typename TSample>
        Direction Evaluate(const TSample &sample) const {
            auto value = feature_.ComputeFeatureValue(sample);
            return threshold_.Evaluate(value);
        }

        const TFeature & GetFeature() const {
            return feature_;
        }

        const TThreshold & GetThreshold() const {
            return threshold_;
        }
        
        template <typename Archive>
        void serialize(Archive &archive, const unsigned int version)
        {
            archive(cereal::make_nvp("feature", feature_));
            archive(cereal::make_nvp("threshold", threshold_));
        }
        
    };

    template <typename TFeature, typename TThreshold>
    class SplitPointCollection {
        std::vector<TFeature> features_;
        std::vector<std::vector<TThreshold> > thresholds_;

    public:
        typedef std::size_t size_type;
        
        void AddSplitPoints(TFeature &&feature, std::vector<TThreshold> &&thresholds) {
            features_.push_back(std::move(feature));
            thresholds_.push_back(std::move(thresholds));
        }

        void AddFeature(TFeature &&feature) {
            features_.push_back(std::move(feature));
            thresholds_.push_back(std::vector<TThreshold>());
        }

        void AddFeature(const TFeature &feature) {
            features_.push_back(feature);
            thresholds_.push_back(std::vector<TThreshold>());
        }

        void AddThreshold(size_type feature_index, const TThreshold &threshold) {
            thresholds_[feature_index].push_back(threshold);
        }

        size_type GetNumOfFeatures() const {
            return features_.size();
        }

        size_type GetNumOfThresholds(size_type feature_index) const {
            return thresholds_[feature_index].size();
        }
        
        const TFeature & GetFeature(size_type feature_index) const {
            return features_[feature_index];
        }

        const TThreshold & GetThreshold(size_type feature_index, size_type threshold_index) const {
            return thresholds_[feature_index][threshold_index];
        }
        
        SplitPoint<TFeature, TThreshold> GetSplitPoint(size_type feature_index, size_type threshold_index) const {
            return SplitPoint<TFeature, TThreshold>(features_[feature_index], thresholds_[feature_index][threshold_index]);
        }

    };

	template <typename TStatistics>
	class SplitStatistics {
        std::vector<std::vector<TStatistics> > left_statistics_collection_;
        std::vector<std::vector<TStatistics> > right_statistics_collection_;

    public:
        typedef typename std::vector<TStatistics>::size_type size_type;

        SplitStatistics()
        {}

		template <typename TFeature, typename TThreshold>
		SplitStatistics(const SplitPointCollection<TFeature, TThreshold> &split_points)
		{
			left_statistics_collection_.resize(split_points.GetNumOfFeatures());
			right_statistics_collection_.resize(split_points.GetNumOfFeatures());
			for (size_type i_f = 0; i_f < split_points.GetNumOfFeatures(); i_f++) {
				left_statistics_collection_[i_f].resize(split_points.GetNumOfThresholds(i_f));
				right_statistics_collection_[i_f].resize(split_points.GetNumOfThresholds(i_f));
			}
		}

        // TODO
        SplitStatistics(std::vector<std::vector<TStatistics> > &&left_statistics_collection, std::vector<std::vector<TStatistics> > &&right_statistics_collection)
			: left_statistics_collection_(left_statistics_collection), right_statistics_collection_(right_statistics_collection)
		{
            if (left_statistics_collection_.size() != right_statistics_collection_.size())
                throw std::runtime_error("The vectors of left and right child statistics must have the same size");
            for (size_type i = 0; i < left_statistics_collection_.size(); i++)
                if (left_statistics_collection_[i].size() != right_statistics_collection_[i].size())
                    throw std::runtime_error("The sub-vectors of left and right child statistics must have the same size");
        }
        
        void AddFeature() {
            left_statistics_collection_.push_back(std::move(std::vector<TStatistics>()));
            right_statistics_collection_.push_back(std::move(std::vector<TStatistics>()));
        }

        void AddThreshold(size_type feature_index) {
            left_statistics_collection_[feature_index].push_back(TStatistics());
            right_statistics_collection_[feature_index].push_back(TStatistics());
        }

        size_type GetNumOfFeatures() const {
            return left_statistics_collection_.size();
        }

        size_type GetNumOfThresholds(size_type feature_index) const {
            return left_statistics_collection_[feature_index].size();
        }

//        std::tuple<const TStatistics &, const TStatistics &> GetSplitStatistics(size_type index) const {
//            return std::make_tuple<const TStatistics &, const TStatistics &>(left_statistics_collection_[index], right_statistics_collection_[index]);
//        }
        
        TStatistics & GetLeftStatistics(size_type feature_index, size_type threshold_index) {
            return left_statistics_collection_[feature_index][threshold_index];
        }
        
        TStatistics & GetRightStatistics(size_type feature_index, size_type threshold_index) {
            return right_statistics_collection_[feature_index][threshold_index];
        }
        
        const TStatistics & GetLeftStatistics(size_type feature_index, size_type threshold_index) const {
            return left_statistics_collection_[feature_index][threshold_index];
        }
        
        const TStatistics & GetRightStatistics(size_type feature_index, size_type threshold_index) const {
            return right_statistics_collection_[feature_index][threshold_index];
        }
        

//		const std::vector<TStatistics> & LeftStatisticsCollection() const {
//			return left_statistics_collection_;
//		}

//		const std::vector<TStatistics> & RightStatisticsCollection() const {
//			return right_statistics_collection_;
//		}

	};

    template <typename TFeature, typename TThreshold, typename TStatistics, typename TSampleIterator, typename TRandomEngine, typename t_size_type = std::size_t>
    class WeakLearner {
	public:
        typedef SplitPoint<TFeature, TThreshold> _SplitPoint;
        typedef SplitPointCollection<TFeature, TThreshold> _SplitPointCollection;
		typedef TStatistics Statistics;
		typedef TSampleIterator SampleIterator;
		typedef t_size_type size_type;
		typedef typename TStatistics::entropy_type entropy_type;

        WeakLearner()
        {}

		virtual ~WeakLearner()
        {}
        
        // TODO: not virtual
		virtual TStatistics ComputeStatistics(TSampleIterator first_sample, TSampleIterator last_sample) const = 0;
        
        // TODO: implement here
        virtual SplitPointCollection<TFeature, TThreshold> SampleSplitPoints(TSampleIterator first_sample, TSampleIterator last_sample, size_type num_of_features, size_type num_of_thresholds, TRandomEngine &rnd_engine) const = 0;

        // TODO: implement here
        virtual SplitStatistics<TStatistics> ComputeSplitStatistics(TSampleIterator first_sample, TSampleIterator last_sample, const SplitPointCollection<TFeature, TThreshold> &split_points) const = 0;

        virtual std::tuple<size_type, size_type, typename TStatistics::entropy_type> FindBestSplitPoint(const TStatistics &current_statistics, const SplitStatistics<TStatistics> &split_statistics) const {
            size_type best_feature_index = 0;
            size_type best_threshold_index = 0;
            entropy_type best_information_gain = -std::numeric_limits<entropy_type>::infinity();
            for (size_type i_f=0; i_f < split_statistics.GetNumOfFeatures(); i_f++) {
                for (size_type i_t=0; i_t < split_statistics.GetNumOfThresholds(i_f); i_t++) {
                    entropy_type information_gain = ComputeInformationGain(current_statistics,
                                                                           split_statistics.GetLeftStatistics(i_f, i_t),
                                                                           split_statistics.GetRightStatistics(i_f, i_t));
                    if (information_gain > best_information_gain) {
                        best_information_gain = information_gain;
                        best_feature_index = i_f;
                        best_threshold_index = i_t;
                    }
                }
            }
            return std::make_tuple(best_feature_index, best_threshold_index, best_information_gain);
        }
        
        virtual entropy_type ComputeInformationGain(const TStatistics &current_statistics, const TStatistics &left_statistics, const TStatistics &right_statistics) const {
            entropy_type current_entropy = current_statistics.Entropy();
            entropy_type left_entropy = left_statistics.Entropy();
            entropy_type right_entropy = right_statistics.Entropy();
            entropy_type information_gain = current_entropy
            - (left_statistics.NumOfSamples() * left_entropy + right_statistics.NumOfSamples() * right_entropy) / static_cast<entropy_type>(current_statistics.NumOfSamples());
            return information_gain;
        }

        TSampleIterator Partition(TSampleIterator first_sample, TSampleIterator last_sample, const SplitPoint<TFeature, TThreshold> &split_point) const {
            TSampleIterator it_left = first_sample;
            TSampleIterator it_right = last_sample - 1;
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
            TSampleIterator it_split;
            if (direction == Direction::LEFT)
                it_split = it_left + 1;
            else
                it_split = it_left;
            
            // check partitioning
            for (TSampleIterator it = first_sample; it != it_split; it++) {
                Direction dir = split_point.Evaluate(*it);
                if (dir != Direction::LEFT)
                    throw std::runtime_error("Samples are not partitioned properly.");
            }
            for (TSampleIterator it = it_split; it != last_sample; it++) {
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
