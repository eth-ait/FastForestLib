#ifndef AITDistributedRandomForest_image_weak_learner_h
#define AITDistributedRandomForest_image_weak_learner_h

#include <tuple>
#include <iostream>

#include <Eigen/Dense>

#include "node.h"
#include "weak_learner.h"
#include "histogram_statistics.h"

namespace AIT {

	template <typename t_data_type = double, typename t_label_type = std::size_t>
	class Image {
	public:
		typedef t_data_type data_type;
		typedef t_label_type label_type;
		typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> DataMatrixType;
		typedef Eigen::Map<DataMatrixType> DataMapType;
		typedef Eigen::Matrix<label_type, Eigen::Dynamic, Eigen::Dynamic> LabelMatrixType;
		typedef Eigen::Map<LabelMatrixType> LabelMapType;

	private:
		DataMapType data_matrix_;
		LabelMapType label_matrix_;

	public:
		Image(const DataMapType &data_matrix, const LabelMapType &label_matrix)
			: data_matrix_(data_matrix), label_matrix_(label_matrix) {}

		Image(DataMapType &data_matrix, LabelMapType &label_matrix)
		: data_matrix_(std::move(data_matrix)), label_matrix_(std::move(label_matrix)) {}

		const DataMapType & GetDataMatrix() const {
			return data_matrix_;
		}

		const LabelMapType & GetLabelMatrix() const {
			return label_matrix_;
		}
	};

	template <typename data_type = double, typename label_type = std::size_t>
	class ImageSample {
		label_type label_;
		std::unique_ptr<Image<data_type, label_type>> image_ptr_;

	public:
		const label_type GetLabel() const {
			return label_;
		}
	};

	template <typename data_type = double>
	class ImageSplitPoint {
	public:
		Direction Evaluate(const ImageSample<data_type> &sample) const {
			return Direction::LEFT;
		}
	};

	template <typename TStatistics, typename TIterator, typename size_type = std::size_t>
	// TODO: ImageSplitPoint template parameter
	class ImageWeakLearner : WeakLearner<ImageSplitPoint<>, TStatistics, TIterator, size_type> {
	public:
		virtual TStatistics ComputeStatistics(TIterator first_sample, TIterator last_sample) const {
			TStatistics statistics;
			for (TIterator sample_it=first_sample; sample_it != last_sample; sample_it++) {
				// TODO
				//statistics.Accumulate(*sample_it);
			}
			return statistics;
		}

		virtual std::vector<ImageSplitPoint<>> SampleSplitPoints(TIterator first_sample, TIterator last_sample, size_type num_of_features, size_type num_of_thresholds) const {
			std::vector<ImageSplitPoint<>> split_points;
			// TODO
			for (size_type i_f=0; i_f < num_of_features; i_f++) {
				for (size_type i_t=0; i_t < num_of_thresholds; i_t++) {
					split_points.push_back(ImageSplitPoint<>());
				}
			}
			return split_points;
		}

		virtual SplitStatistics<TStatistics> ComputeSplitStatistics(TIterator first_sample, TIterator last_sample, const std::vector<ImageSplitPoint<>> &split_points) const {
			std::vector<TStatistics> left_statistics_collection;
			std::vector<TStatistics> right_statistics_collection;
			for (auto split_point_it = split_points.cbegin(); split_point_it != split_points.cend(); split_point_it++) {
				TStatistics left_statistics;
				TStatistics right_statistics;
				for (TIterator sample_it=first_sample; sample_it != last_sample; sample_it++) {
					Direction direction = split_point_it->Evaluate(*sample_it);
					if (direction == Direction::LEFT)
						left_statistics.Accumulate(*sample_it);
					else
						right_statistics.Accumulate(*sample_it);
				}
				left_statistics_collection.push_back(std::move(left_statistics));
				right_statistics_collection.push_back(std::move(right_statistics));
			}
			return SplitStatistics<TStatistics>(std::move(left_statistics_collection), std::move(right_statistics_collection));
		}

		virtual std::tuple<size_type, entropy_type> FindBestSplitPointIndex(const TStatistics &current_statistics, const std::vector<TStatistics> &split_statistics) const {
			// TODO
			size_type best_split_point_index = 0;
			entropy_type best_split_point_entropy = split_statistics[0].Entropy();
	//          for (size_type i=1; i < )
	//          std::tuple<size_type, value_type> best_split_point_tuple;
			return std::make_tuple(best_split_point_index, best_split_point_entropy);
		}

		virtual TIterator Partition(TIterator it_left, TIterator it_right, const ImageSplitPoint<> &best_split_point) const {
			return it_left;
		}
	};

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
//  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}

#endif
