#ifndef AITDistributedRandomForest_image_weak_learner_h
#define AITDistributedRandomForest_image_weak_learner_h

#include <tuple>
#include <iostream>
#include <random>
#include <cmath>

#include <Eigen/Dense>

#include "node.h"
#include "weak_learner.h"
#include "histogram_statistics.h"

namespace AIT {
    
    class ImageWeakLearnerParameters {
    public:
        int FeatureOffsetXRangeLow() const {
            return 3;
        }
        int FeatureOffsetXRangeHigh() const {
            return 15;
        }
        int FeatureOffsetYRangeLow() const {
            return 3;
        }
        int FeatureOffsetYRangeHigh() const {
            return 15;
        }
        double ThresholdRangeLow() const {
            return -300.0;
        }
        double ThresholdRangeHigh() const {
            return +300.0;
        }
    };

	template <typename t_data_type = double, typename t_label_type = std::size_t>
	class Image {
	public:
		typedef t_data_type data_type;
		typedef t_label_type label_type;
		typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> DataMatrixType;
//		typedef Eigen::Map<DataMatrixType> DataMapType;
		typedef Eigen::Matrix<label_type, Eigen::Dynamic, Eigen::Dynamic> LabelMatrixType;
//		typedef Eigen::Map<LabelMatrixType> LabelMapType;

	private:
		DataMatrixType data_matrix_;
		LabelMatrixType label_matrix_;

        void CheckEqualDimensions(const DataMatrixType &data_matrix, const LabelMatrixType &label_matrix) const {
            if (data_matrix.rows() != label_matrix.rows() || data_matrix.cols() != label_matrix.cols())
                throw std::runtime_error("The data and label matrix must have the same dimension.");
        }
	public:
		Image(const DataMatrixType &data_matrix, const LabelMatrixType &label_matrix)
        : data_matrix_(data_matrix), label_matrix_(label_matrix)
        {
            CheckEqualDimensions(data_matrix, label_matrix);
        }

		Image(DataMatrixType &&data_matrix, LabelMatrixType &&label_matrix)
        : data_matrix_(std::move(data_matrix)), label_matrix_(std::move(label_matrix))
        {
            CheckEqualDimensions(data_matrix, label_matrix);
        }

		const DataMatrixType & GetDataMatrix() const {
			return data_matrix_;
		}

		const LabelMatrixType & GetLabelMatrix() const {
			return label_matrix_;
		}
        
        int Width() const {
            return data_matrix_.rows();
        }

        int Height() const {
            return data_matrix_.cols();
        }
	};

	template <typename t_data_type = double, typename t_label_type = std::size_t>
	class ImageSample {
    public:
        typedef t_data_type data_type;
		typedef t_label_type label_type;
	private:
        const Image<data_type, label_type> *image_ptr_;
        int x_;
        int y_;

	public:
        ImageSample(const Image<data_type, label_type> *image_ptr, int x, int y)
        : image_ptr_(image_ptr), x_(x), y_(y) {}

        ImageSample(const ImageSample &other)
        : image_ptr_(other.image_ptr_), x_(other.x_), y_(other.y_) {}

		const label_type GetLabel() const {
			return image_ptr_->GetLabelMatrix()(x_, y_);
		}

        const Image<data_type, label_type> & GetImage() const {
            return *image_ptr_;
        }

        int GetX() const {
            return x_;
        }

        int GetY() const {
            return y_;
        }
	};
    
    template <typename data_type = double, typename label_type = std::size_t, typename offset_type = int>
	class ImageSplitPoint {
        offset_type offset_x1_;
        offset_type offset_y1_;
        offset_type offset_x2_;
        offset_type offset_y2_;
        data_type threshold_;
        
        inline data_type ComputePixelDifference(const ImageSample<data_type, label_type> &sample) const {
            data_type pixel1_value = ComputePixelValue(sample, offset_x1_, offset_y1_);
            data_type pixel2_value = ComputePixelValue(sample, offset_x2_, offset_y2_);
            return  pixel1_value - pixel2_value;
        }
        
        inline data_type ComputePixelValue(const ImageSample<data_type, label_type> &sample, offset_type offset_x, offset_type offset_y) const {
            const Image<data_type, label_type> &image = sample.GetImage();
            offset_type x = sample.GetX();
            offset_type y = sample.GetY();
            data_type pixel_value;
            if (x + offset_x < 0 || x + offset_x >= image.Width() || y + offset_y < 0 || y + offset_y >= image.Height())
                pixel_value = 0;
            else
                pixel_value = image.GetDataMatrix()(x + offset_x, y + offset_y);
            return pixel_value;
        }

    public:
        ImageSplitPoint() {}

        ImageSplitPoint(offset_type offset_x1, offset_type offset_y1, offset_type offset_x2, offset_type offset_y2, data_type threshold)
        : offset_x1_(offset_x1), offset_y1_(offset_y1), offset_x2_(offset_x2), offset_y2_(offset_y2), threshold_(threshold) {}

		Direction Evaluate(const ImageSample<data_type, label_type> &sample) const {
            data_type pixel_difference = ComputePixelDifference(sample);
            if (pixel_difference < threshold_)
                return Direction::LEFT;
            else
                return Direction::RIGHT;
        }

	};

    template <typename TStatistics, typename TIterator, typename TRandomEngine, typename size_type = std::size_t>
	// TODO: ImageSplitPoint template parameter
	class ImageWeakLearner : public WeakLearner<ImageSplitPoint<>, TStatistics, TIterator, TRandomEngine, size_type> {
        typedef WeakLearner<ImageSplitPoint<>, TStatistics, TIterator, TRandomEngine, size_type> BaseType;

        const ImageWeakLearnerParameters parameters_;

	public:
        typedef typename TStatistics::entropy_type entropy_type;

        ImageWeakLearner(const ImageWeakLearnerParameters &parameters)
        : parameters_(parameters)
        {}

		virtual ~ImageWeakLearner() {}

		virtual TStatistics ComputeStatistics(TIterator first_sample, TIterator last_sample) const {
			TStatistics statistics;
			for (TIterator sample_it=first_sample; sample_it != last_sample; sample_it++) {
				statistics.Accumulate(*sample_it);
			}
			return statistics;
		}

		virtual std::vector<ImageSplitPoint<> > SampleSplitPoints(TIterator first_sample, TIterator last_sample, size_type num_of_features, size_type num_of_thresholds, TRandomEngine &rnd_engine) const {
			std::vector<ImageSplitPoint<> > split_points;
            // TODO: Seed with parameter value
            // TOOD: Image width?

            // TODO: Fix discrete distributions
            int offset_x_range_low = parameters_.FeatureOffsetXRangeLow();
            int offset_x_range_high = parameters_.FeatureOffsetXRangeHigh();
            std::vector<int> offsets_x;
            for (int offset_x=offset_x_range_low; offset_x <= offset_x_range_high; offset_x++) {
                offsets_x.push_back(-offset_x);
                offsets_x.push_back(+offset_x);
            }
            std::uniform_int_distribution<int> offset_x_distribution(0, offsets_x.size());

            int offset_y_range_low = parameters_.FeatureOffsetYRangeLow();
            int offset_y_range_high = parameters_.FeatureOffsetYRangeHigh();
            std::vector<int> offsets_y;
            for (int offset_y=offset_y_range_low; offset_y <= offset_y_range_high; offset_y++) {
                offsets_y.push_back(-offset_y);
                offsets_y.push_back(+offset_y);
            }
            std::uniform_int_distribution<int> offset_y_distribution(0, offsets_y.size());

            double threshold_range_low = parameters_.ThresholdRangeLow();
            double threshold_range_high = parameters_.ThresholdRangeHigh();
            std::uniform_real_distribution<double> threshold_distribution(threshold_range_low, threshold_range_high);

            for (size_type i_f=0; i_f < num_of_features; i_f++) {
                int offset_x1 = offsets_x[offset_x_distribution(rnd_engine)];
                int offset_y1 = offsets_y[offset_y_distribution(rnd_engine)];
                int offset_x2 = offsets_x[offset_x_distribution(rnd_engine)];
                int offset_y2 = offsets_y[offset_y_distribution(rnd_engine)];
				for (size_type i_t=0; i_t < num_of_thresholds; i_t++) {
                    double threshold = threshold_distribution(rnd_engine);
                    ImageSplitPoint<> split_point(offset_x1, offset_y1, offset_x2, offset_y2, threshold);
					split_points.push_back(split_point);
				}
			}
			return split_points;
		}

		virtual SplitStatistics<TStatistics> ComputeSplitStatistics(TIterator first_sample, TIterator last_sample, const std::vector<ImageSplitPoint<> > &split_points) const {
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

		virtual std::tuple<size_type, entropy_type> FindBestSplitPoint(const TStatistics &current_statistics, const SplitStatistics<TStatistics> &split_statistics) const {
			size_type best_index = 0;
            entropy_type best_information_gain = -std::numeric_limits<entropy_type>::infinity();
            for (size_type i=0; i < split_statistics.Count(); i++) {
                entropy_type information_gain = ComputeInformationGain(current_statistics, split_statistics.GetLeftStatistics(i), split_statistics.GetRightStatistics(i));
                if (information_gain > best_information_gain) {
                    best_information_gain = information_gain;
                    best_index = i;
                }
            }
            return std::make_tuple(best_index, best_information_gain);
		}

        entropy_type ComputeInformationGain(const TStatistics &current_statistics, const TStatistics &left_statistics, const TStatistics &right_statistics) const {
            entropy_type current_entropy = current_statistics.Entropy();
            entropy_type left_entropy = left_statistics.Entropy();
            entropy_type right_entropy = right_statistics.Entropy();
            entropy_type information_gain = current_entropy
                - (left_statistics.NumOfSamples() * left_entropy + right_statistics.NumOfSamples() * right_entropy) / static_cast<entropy_type>(current_statistics.NumOfSamples());
            return information_gain;
        }
	};

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
//  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}

#endif
