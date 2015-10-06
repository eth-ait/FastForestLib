#pragma once

#include <tuple>
#include <iostream>
#include <random>
#include <cmath>

#include <Eigen/Dense>

#include "ait.h"
#include "node.h"
#include "weak_learner.h"
#include "histogram_statistics.h"

namespace ait
{
 
using pixel_type = std::int16_t;
using offset_type = std::int16_t;
using label_type = std::int16_t;

// TODO: Lowercase
class ImageWeakLearnerParameters
{
public:
    int num_of_thresholds() const
    {
        return 10;
    }
    int num_of_features() const
    {
        return 10;
    }
    offset_type feature_offset_x_range_low() const
    {
        return 3;
    }
    offset_type feature_offset_x_range_high() const
    {
        return 15;
    }
    offset_type feature_offset_y_range_low() const
    {
        return 3;
    }
    offset_type feature_offset_y_range_high() const
    {
        return 15;
    }
    scalar_type threshold_range_low() const
    {
        return -300.0;
    }
    scalar_type threshold_range_high() const
    {
        return +300.0;
    }
};

class Image
{
public:
    using DataMatrixType = Eigen::Matrix<pixel_type, Eigen::Dynamic, Eigen::Dynamic>;
    using LabelMatrixType = Eigen::Matrix<label_type, Eigen::Dynamic, Eigen::Dynamic>;
    // TODO
//    using DataMapType = Eigen::Map<DataMatrixType>;
//    using LabelMapType = Eigen::Map<LabelMatrixType>;

private:
    DataMatrixType data_matrix_;
    LabelMatrixType label_matrix_;

    void check_equal_dimensions(const DataMatrixType &data_matrix, const LabelMatrixType &label_matrix) const
    {
        if (data_matrix.rows() != label_matrix.rows() || data_matrix.cols() != label_matrix.cols())
            throw std::runtime_error("The data and label matrix must have the same dimension.");
    }
public:
    Image(const DataMatrixType &data_matrix, const LabelMatrixType &label_matrix)
    : data_matrix_(data_matrix), label_matrix_(label_matrix)
    {
        check_equal_dimensions(data_matrix, label_matrix);
    }

    Image(DataMatrixType &&data_matrix, LabelMatrixType &&label_matrix)
    : data_matrix_(std::move(data_matrix)), label_matrix_(std::move(label_matrix))
    {
        check_equal_dimensions(data_matrix, label_matrix);
    }

    const DataMatrixType & get_data_matrix() const
    {
        return data_matrix_;
    }

    const LabelMatrixType & get_label_matrix() const
    {
        return label_matrix_;
    }
    
    size_type width() const
    {
        return data_matrix_.rows();
    }

    size_type height() const
    {
        return data_matrix_.cols();
    }
};

class ImageSample
{
private:
    const Image *image_ptr_;
    offset_type x_;
    offset_type y_;

public:
    ImageSample(const Image *image_ptr, offset_type x, offset_type y)
    : image_ptr_(image_ptr), x_(x), y_(y)
    {}

    ImageSample(const ImageSample &other)
    : image_ptr_(other.image_ptr_), x_(other.x_), y_(other.y_)
    {}

    const label_type get_label() const
    {
        return image_ptr_->get_label_matrix()(x_, y_);
    }

    const Image & get_image() const
    {
        return *image_ptr_;
    }

    offset_type get_x() const
    {
        return x_;
    }

    offset_type get_y() const
    {
        return y_;
    }
};

class ImageSplitPoint {
    offset_type offset_x1_;
    offset_type offset_y1_;
    offset_type offset_x2_;
    offset_type offset_y2_;
    scalar_type threshold_;
    
    scalar_type compute_pixel_difference(const ImageSample &sample) const {
        pixel_type pixel1_value = compute_pixel_value(sample, offset_x1_, offset_y1_);
        pixel_type pixel2_value = compute_pixel_value(sample, offset_x2_, offset_y2_);
        return pixel1_value - pixel2_value;
    }
    
    scalar_type compute_pixel_value(const ImageSample &sample, offset_type offset_x, offset_type offset_y) const {
        const Image &image = sample.get_image();
        offset_type x = sample.get_x();
        offset_type y = sample.get_y();
        pixel_type pixel_value;
        if (x + offset_x < 0 || x + offset_x >= image.width() || y + offset_y < 0 || y + offset_y >= image.height())
            pixel_value = 0;
        else
            pixel_value = image.get_data_matrix()(x + offset_x, y + offset_y);
        return pixel_value;
    }

public:
    ImageSplitPoint() {}

    ImageSplitPoint(offset_type offset_x1, offset_type offset_y1, offset_type offset_x2, offset_type offset_y2, scalar_type threshold)
    : offset_x1_(offset_x1), offset_y1_(offset_y1), offset_x2_(offset_x2), offset_y2_(offset_y2), threshold_(threshold) {}

    // TODO
    /*pixel_type compute_split_point_value(const ImageSample &sample) const {
        return compute_pixel_difference(sample);
    }*/

    Direction evaluate(const ImageSample &sample) const
    {
        pixel_type pixel_difference = compute_pixel_difference(sample);
        return evaluate(pixel_difference);
    }

    Direction evaluate(pixel_type value) const
    {
        if (value < threshold_)
            return Direction::LEFT;
        else
            return Direction::RIGHT;
    }
    
    scalar_type get_offset_x1() const
    {
        return offset_x1_;
    }

    scalar_type get_offset_y1() const
    {
        return offset_y1_;
    }
    
    scalar_type get_offset_x2() const
    {
        return offset_x2_;
    }
    
    scalar_type get_offset_y2() const
    {
        return offset_y2_;
    }

    scalar_type get_threshold() const
    {
        return threshold_;
    }

    template <typename Archive>
    void serialize(Archive &archive, const unsigned int version)
    {
#ifdef SERIALIZE_WITH_BOOST
        archive & offset_x1_;
        archive & offset_y1_;
        archive & offset_x2_;
        archive & offset_y2_;
        archive & threshold_;
#else
        archive(cereal::make_nvp("offset_x1", offset_x1_));
        archive(cereal::make_nvp("offset_y1", offset_y1_));
        archive(cereal::make_nvp("offset_x2", offset_x2_));
        archive(cereal::make_nvp("offset_y2", offset_y2_));
        archive(cereal::make_nvp("threshold", threshold_));
#endif
    }
};

template <typename TStatisticsFactory, typename TSampleIterator, typename TRandomEngine = std::mt19937_64>
class ImageWeakLearner : public WeakLearner<ImageSplitPoint, TStatisticsFactory, TSampleIterator, TRandomEngine>
{
    using BaseType = WeakLearner<ImageSplitPoint, TStatisticsFactory, TSampleIterator, TRandomEngine>;

    const ImageWeakLearnerParameters parameters_;

public:
    using StatisticsT = typename BaseType::StatisticsT;

    ImageWeakLearner(const ImageWeakLearnerParameters &parameters, const TStatisticsFactory &statistics_factory)
    : BaseType(statistics_factory), parameters_(parameters)
    {}

    ~ImageWeakLearner() {}

    virtual std::vector<ImageSplitPoint> sample_split_points(TSampleIterator first_sample, TSampleIterator last_sample, TRandomEngine &rnd_engine) const
    {
        std::vector<ImageSplitPoint> split_points;
        // TODO: Seed with parameter value
        // TOOD: Image width?

        // TODO: Fix discrete offset distributions
        offset_type offset_x_range_low = parameters_.feature_offset_x_range_low();
        offset_type offset_x_range_high = parameters_.feature_offset_x_range_high();
        std::vector<offset_type> offsets_x;
        for (offset_type offset_x=offset_x_range_low; offset_x <= offset_x_range_high; offset_x++) {
            offsets_x.push_back(-offset_x);
            offsets_x.push_back(+offset_x);
        }
        std::uniform_int_distribution<offset_type> offset_x_distribution(0, offsets_x.size() - 1);

        offset_type offset_y_range_low = parameters_.feature_offset_y_range_low();
        offset_type offset_y_range_high = parameters_.feature_offset_y_range_high();
        std::vector<offset_type> offsets_y;
        for (offset_type offset_y=offset_y_range_low; offset_y <= offset_y_range_high; offset_y++) {
            offsets_y.push_back(-offset_y);
            offsets_y.push_back(+offset_y);
        }
        std::uniform_int_distribution<offset_type> offset_y_distribution(0, offsets_y.size() - 1);

        scalar_type threshold_range_low = parameters_.threshold_range_low();
        scalar_type threshold_range_high = parameters_.threshold_range_high();
        std::uniform_real_distribution<scalar_type> threshold_distribution(threshold_range_low, threshold_range_high);

        for (size_type i_f=0; i_f < parameters_.num_of_features(); i_f++) {
            offset_type offset_x1 = offsets_x[offset_x_distribution(rnd_engine)];
            offset_type offset_y1 = offsets_y[offset_y_distribution(rnd_engine)];
            offset_type offset_x2 = offsets_x[offset_x_distribution(rnd_engine)];
            offset_type offset_y2 = offsets_y[offset_y_distribution(rnd_engine)];
            for (size_type i_t=0; i_t < parameters_.num_of_thresholds(); i_t++) {
                scalar_type threshold = threshold_distribution(rnd_engine);
                ImageSplitPoint split_point(offset_x1, offset_y1, offset_x2, offset_y2, threshold);
                split_points.push_back(split_point);
            }
        }
        return split_points;
    }

    virtual SplitStatistics<StatisticsT> compute_split_statistics(TSampleIterator first_sample, TSampleIterator last_sample, const std::vector<ImageSplitPoint> &split_points) const
    {
        // we create statistics for all features and thresholds here so that we can easily parallelize the loop below
        SplitStatistics<StatisticsT> split_statistics(split_points.size(), this->statistics_factory_);
        // TODO: Parallelize
        //#pragma omp parallel for
        // we have to use signed int here because of OpenMP < 3.0
        for (TSampleIterator sample_it = first_sample; sample_it != last_sample; sample_it++)
        {
            for (size_type i = 0; i < split_points.size(); i++)
            {
                Direction direction = split_points[i].evaluate(*sample_it);
                if (direction == Direction::LEFT)
                    split_statistics.get_left_statistics(i).lazy_accumulate(*sample_it);
                else
                    split_statistics.get_right_statistics(i).lazy_accumulate(*sample_it);
            }
        }

        for (size_type i = 0; i < split_points.size(); i++) {
            split_statistics.get_left_statistics(i).finish_lazy_accumulation();
            split_statistics.get_right_statistics(i).finish_lazy_accumulation();
        }
        return split_statistics;
    }

};

//friend std::ostream & operator<<(std::ostream &os, const SplitPoint &splitPoint);
//virtual void writeToStream(std::ostream &os);
//std::ostream & operator<<(std::ostream &os, const SplitPoint &split_point);

}
