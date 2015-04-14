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
		typedef EIGEN_DEFAULT_DENSE_INDEX_TYPE size_type;
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
        
		size_type Width() const {
            return data_matrix_.rows();
        }

		size_type Height() const {
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

//    template <typename data_type = double, typename label_type = std::size_t, typename offset_type = int>
//	class ImageSplitPoint {
//        offset_type offset_x1_;
//        offset_type offset_y1_;
//        offset_type offset_x2_;
//        offset_type offset_y2_;
//        data_type threshold_;
//        
//        data_type ComputePixelDifference(const ImageSample<data_type, label_type> &sample) const {
//            data_type pixel1_value = ComputePixelValue(sample, offset_x1_, offset_y1_);
//            data_type pixel2_value = ComputePixelValue(sample, offset_x2_, offset_y2_);
//            return  pixel1_value - pixel2_value;
//        }
//        
//        data_type ComputePixelValue(const ImageSample<data_type, label_type> &sample, offset_type offset_x, offset_type offset_y) const {
//            const Image<data_type, label_type> &image = sample.GetImage();
//            offset_type x = sample.GetX();
//            offset_type y = sample.GetY();
//            data_type pixel_value;
//            if (x + offset_x < 0 || x + offset_x >= image.Width() || y + offset_y < 0 || y + offset_y >= image.Height())
//                pixel_value = 0;
//            else
//                pixel_value = image.GetDataMatrix()(x + offset_x, y + offset_y);
//            return pixel_value;
//        }
//
//    public:
//        ImageSplitPoint() {}
//
//        ImageSplitPoint(offset_type offset_x1, offset_type offset_y1, offset_type offset_x2, offset_type offset_y2, data_type threshold)
//        : offset_x1_(offset_x1), offset_y1_(offset_y1), offset_x2_(offset_x2), offset_y2_(offset_y2), threshold_(threshold) {}
//
//        data_type ComputeSplitPointValue(const ImageSample<data_type, label_type> &sample) const {
//            return ComputePixelDifference(sample);
//        }
//
//		Direction Evaluate(const ImageSample<data_type, label_type> &sample) const {
//            data_type pixel_difference = ComputePixelDifference(sample);
//            if (pixel_difference < threshold_)
//                return Direction::LEFT;
//            else
//                return Direction::RIGHT;
//        }
//
//        Direction Evaluate(data_type value) const {
//            if (value < threshold_)
//                return Direction::LEFT;
//            else
//                return Direction::RIGHT;
//        }
//
//        data_type GetThreshold() const {
//            return threshold_;
//        }
//
//	};
    
    template <typename data_type = double>
    class Threshold {
        data_type threshold_;

    public:
        Threshold() {}
        
        Threshold(data_type threshold)
        : threshold_(threshold) {}

        data_type GetThreshold() const {
            return threshold_;
        }

		Direction Evaluate(data_type value) const {
            if (value < threshold_)
                return Direction::LEFT;
            else
                return Direction::RIGHT;
        }
        
        template <typename Archive>
        void serialize(Archive &archive, const unsigned int version)
        {
            archive(cereal::make_nvp("threshold", threshold_));
        }
        
    };

    template <typename data_type = double, typename label_type = std::size_t, typename offset_type = int>
    class ImageFeature {
        offset_type offset_x1_;
        offset_type offset_y1_;
        offset_type offset_x2_;
        offset_type offset_y2_;
        
        data_type ComputePixelDifference(const ImageSample<data_type, label_type> &sample) const {
            data_type pixel1_value = ComputePixelValue(sample, offset_x1_, offset_y1_);
            data_type pixel2_value = ComputePixelValue(sample, offset_x2_, offset_y2_);
            return  pixel1_value - pixel2_value;
        }
        
        data_type ComputePixelValue(const ImageSample<data_type, label_type> &sample, offset_type offset_x, offset_type offset_y) const {
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
        ImageFeature() {}
        
        ImageFeature(offset_type offset_x1, offset_type offset_y1, offset_type offset_x2, offset_type offset_y2)
        : offset_x1_(offset_x1), offset_y1_(offset_y1), offset_x2_(offset_x2), offset_y2_(offset_y2)
        {}
        
		data_type ComputeFeatureValue(const ImageSample<data_type, label_type> &sample) const {
            return ComputePixelDifference(sample);
        }
        
        template <typename Archive>
        void serialize(Archive &archive, const unsigned int version)
        {
            archive(cereal::make_nvp("offset_x1", offset_x1_));
            archive(cereal::make_nvp("offset_y1", offset_y1_));
            archive(cereal::make_nvp("offset_x2", offset_x2_));
            archive(cereal::make_nvp("offset_y2", offset_y2_));
        }

    };
    
    template <typename TStatistics, typename TIterator, typename TRandomEngine, typename size_type = std::size_t>
	class ImageWeakLearner : public WeakLearner<ImageFeature<>, Threshold<>, TStatistics, TIterator, TRandomEngine, size_type> {
        typedef WeakLearner<ImageFeature<>, Threshold<>, TStatistics, TIterator, TRandomEngine, size_type> BaseType;

        const ImageWeakLearnerParameters parameters_;

	public:
        typedef typename TStatistics::entropy_type entropy_type;

        ImageWeakLearner(const ImageWeakLearnerParameters &parameters)
        : parameters_(parameters)
        {}

		~ImageWeakLearner() {}

		virtual TStatistics ComputeStatistics(TIterator first_sample, TIterator last_sample) const {
			TStatistics statistics;
			for (TIterator sample_it=first_sample; sample_it != last_sample; sample_it++) {
				statistics.Accumulate(*sample_it);
			}
			return statistics;
		}

		virtual SplitPointCollection<ImageFeature<>, Threshold<> > SampleSplitPoints(TIterator first_sample, TIterator last_sample, size_type num_of_features, size_type num_of_thresholds, TRandomEngine &rnd_engine) const {
			typedef typename std::vector<int>::size_type vec_size_type;
            SplitPointCollection<ImageFeature<>, Threshold<> > split_points;
//			std::vector<ImageSplitPoint<> > split_points;
            // TODO: Seed with parameter value
            // TOOD: Image width?

            // TODO: Fix discrete offset distributions
            int offset_x_range_low = parameters_.FeatureOffsetXRangeLow();
            int offset_x_range_high = parameters_.FeatureOffsetXRangeHigh();
            std::vector<int> offsets_x;
            for (int offset_x=offset_x_range_low; offset_x <= offset_x_range_high; offset_x++) {
                offsets_x.push_back(-offset_x);
                offsets_x.push_back(+offset_x);
            }
            std::uniform_int_distribution<vec_size_type> offset_x_distribution(0, offsets_x.size() - 1);

            int offset_y_range_low = parameters_.FeatureOffsetYRangeLow();
            int offset_y_range_high = parameters_.FeatureOffsetYRangeHigh();
            std::vector<int> offsets_y;
            for (int offset_y=offset_y_range_low; offset_y <= offset_y_range_high; offset_y++) {
                offsets_y.push_back(-offset_y);
                offsets_y.push_back(+offset_y);
            }
			std::uniform_int_distribution<vec_size_type> offset_y_distribution(0, offsets_y.size() - 1);

            double threshold_range_low = parameters_.ThresholdRangeLow();
            double threshold_range_high = parameters_.ThresholdRangeHigh();
            std::uniform_real_distribution<double> threshold_distribution(threshold_range_low, threshold_range_high);

            for (size_type i_f=0; i_f < num_of_features; i_f++) {
                int offset_x1 = offsets_x[offset_x_distribution(rnd_engine)];
                int offset_y1 = offsets_y[offset_y_distribution(rnd_engine)];
                int offset_x2 = offsets_x[offset_x_distribution(rnd_engine)];
                int offset_y2 = offsets_y[offset_y_distribution(rnd_engine)];
                ImageFeature<> feature(offset_x1, offset_y1, offset_x2, offset_y2);
                split_points.AddFeature(feature);
				for (size_type i_t=0; i_t < num_of_thresholds; i_t++) {
                    double threshold = threshold_distribution(rnd_engine);
                    split_points.AddThreshold(i_f, threshold);
				}
			}
			return split_points;
		}

        virtual SplitStatistics<TStatistics> ComputeSplitStatistics(TIterator first_sample, TIterator last_sample, const SplitPointCollection<ImageFeature<>, Threshold<> > &split_points) const {
			// we create statistics for all features and thresholds here so that we can easily parallelize the loop below
			SplitStatistics<TStatistics> split_statistics(split_points);
			//#pragma omp parallel for
			// we have to use signed int here because of OpenMP < 3.0
			for (int i_f = 0; i_f < split_points.GetNumOfFeatures(); i_f++) {
                const ImageFeature<> &feature = split_points.GetFeature(i_f);
                for (TIterator sample_it=first_sample; sample_it != last_sample; sample_it++) {
                    double value = feature.ComputeFeatureValue(*sample_it);
                    for (size_type i_t = 0; i_t < split_points.GetNumOfThresholds(i_f); i_t++) {
                        Direction direction = split_points.GetThreshold(i_f, i_t).Evaluate(value);
                        if (direction == Direction::LEFT)
                            split_statistics.GetLeftStatistics(i_f, i_t).LazyAccumulate(*sample_it);
                        else
							split_statistics.GetRightStatistics(i_f, i_t).LazyAccumulate(*sample_it);
                    }
				}
            }
            for (int i_f = 0; i_f < split_points.GetNumOfFeatures(); i_f++) {
                for (size_type i_t = 0; i_t < split_points.GetNumOfThresholds(i_f); i_t++) {
                    split_statistics.GetLeftStatistics(i_f, i_t).FinishLazyAccumulation();
                    split_statistics.GetRightStatistics(i_f, i_t).FinishLazyAccumulation();
                }
            }
            return split_statistics;
		}

    };

      //friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
      //virtual void writeToStream(std::ostream &os);
//  std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}

#endif
