//
//  image_weak_learner.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

#include <tuple>
#include <iostream>
#include <random>
#include <cmath>

#include <Eigen/Dense>
#include <CImg.h>

#include "ait.h"
#include "logger.h"
#include "node.h"
#include "weak_learner.h"
#include "histogram_statistics.h"
#include "bagging_wrapper.h"

namespace ait
{

using pixel_type = std::int16_t;
using offset_type = std::int16_t;
using label_type = std::int16_t;

class ImageWeakLearnerParameters
{
public:
    double samples_per_image_fraction = 0.01;
    double bagging_fraction = 0.2;
    int num_of_thresholds = 10;
    int num_of_features = 10;
    offset_type feature_offset_x_range_low = 3;
    offset_type feature_offset_x_range_high = 15;
    offset_type feature_offset_y_range_low = 3;
    offset_type feature_offset_y_range_high = 15;
    scalar_type threshold_range_low = -300.0;
    scalar_type threshold_range_high = +300;
};

template <typename TPixel = pixel_type>
class Image
{
public:
    using PixelT = TPixel;
    using DataMatrixType = Eigen::Matrix<TPixel, Eigen::Dynamic, Eigen::Dynamic>;
    using LabelMatrixType = Eigen::Matrix<TPixel, Eigen::Dynamic, Eigen::Dynamic>;

private:
    DataMatrixType data_matrix_;
    LabelMatrixType label_matrix_;

    void check_equal_dimensions(const DataMatrixType& data_matrix, const LabelMatrixType& label_matrix) const
    {
        if (data_matrix.rows() != label_matrix.rows() || data_matrix.cols() != label_matrix.cols())
            throw std::runtime_error("The data and label matrix must have the same dimension.");
    }

public:
    explicit Image(const DataMatrixType& data_matrix, const LabelMatrixType& label_matrix)
    : data_matrix_(data_matrix), label_matrix_(label_matrix)
    {
        check_equal_dimensions(data_matrix, label_matrix);
    }

    explicit Image(DataMatrixType&& data_matrix, LabelMatrixType&& label_matrix)
    : data_matrix_(std::move(data_matrix)), label_matrix_(std::move(label_matrix))
    {
        check_equal_dimensions(data_matrix, label_matrix);
    }

    const DataMatrixType& get_data_matrix() const
    {
        return data_matrix_;
    }

    const LabelMatrixType& get_label_matrix() const
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

    static Image load_from_files(const std::string& data_filename, const std::string& label_filename)
    {
        cimg_library::CImg<TPixel> data_image(data_filename.c_str());
        cimg_library::CImg<TPixel> label_image(label_filename.c_str());
        int width = data_image.width();
        int height = data_image.height();
        int depth = data_image.depth();
        int spectrum = data_image.spectrum();
        // TODO: Show error if this happens
        assert(width == label_image.width());
        assert(height == label_image.height());
        assert(depth == label_image.depth());
        assert(spectrum == label_image.spectrum());
        assert(depth == 1);
        assert(spectrum == 1);
        DataMatrixType data(width, height);
        LabelMatrixType label(width, height);
        for (int w = 0; w < width; ++w)
        {
            for (int h = 0; h < height; ++h)
            {
                data(w, h) = data_image(w, h, 0, 0, 0, 0);
                label(w, h) = label_image(w, h, 0, 0, 0, 0);
            }
        }
        return Image(data, label);
    }
};

template <typename TPixel = pixel_type>
class ImageSample
{
public:
    using PixelT = TPixel;

    friend void swap(ImageSample& a, ImageSample& b) {
        using std::swap;
        std::swap(a.image_ptr_, b.image_ptr_);
        std::swap(a.x_, b.x_);
        std::swap(a.y_, b.y_);
    }

    ImageSample(const Image<TPixel>* image_ptr, offset_type x, offset_type y)
    : image_ptr_(image_ptr), x_(x), y_(y)
    {}

    ImageSample(const ImageSample& other)
    : image_ptr_(other.image_ptr_), x_(other.x_), y_(other.y_)
    {}

    const label_type get_label() const
    {
        return image_ptr_->get_label_matrix()(x_, y_);
    }

    const Image<TPixel>& get_image() const
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

private:
    const Image<TPixel>* image_ptr_;
    offset_type x_;
    offset_type y_;
};

template <typename TRandomEngine, typename TPixel = pixel_type>
class ImageSampleProvider : public BaggingSampleProvider<TRandomEngine, ImageSample<TPixel>>
{
    using SampleT = ImageSample<TPixel>;

public:
    ImageSampleProvider(const std::vector<Image<TPixel>>& images, const ImageWeakLearnerParameters& parameters)
    : images_(images), parameters_(parameters)
    {}

    virtual std::vector<SampleT> get_sample_bag(TRandomEngine& rnd_engine) const
    {
        log_info(false) << "Creating sample bag ...";
        std::vector<SampleT> samples;
        int images_per_bag = std::round(parameters_.bagging_fraction * images_.size());
        std::uniform_int_distribution<> image_dist(0, images_.size() - 1);
        for (int i = 0; i < images_per_bag; i++)
        {
            int image_index = image_dist(rnd_engine);
            const Image<TPixel>& image = images_[image_index];
            int num_of_samples_per_image = std::round(parameters_.samples_per_image_fraction * image.width() * image.height());
            std::uniform_int_distribution<> x_dist(0, image.width() - 1);
            std::uniform_int_distribution<> y_dist(0, image.height() - 1);
            for (int j = 0; j < num_of_samples_per_image; j++)
            {
                int x = x_dist(rnd_engine);
                int y = y_dist(rnd_engine);
                ImageSample<TPixel> sample(&image, x, y);
                samples.push_back(std::move(sample));
            }
        }
        log_info(true) << "Done";
        return samples;
    }

private:
    const std::vector<Image<TPixel>>& images_;
    const ImageWeakLearnerParameters parameters_;
};

template <typename TPixel = pixel_type>
class ImageSplitPoint {
public:
    using PixelT = TPixel;

    ImageSplitPoint()
    : offset_x1_(0), offset_y1_(0), offset_x2_(0), offset_y2_(0), threshold_(0)
    {}

    ImageSplitPoint(offset_type offset_x1, offset_type offset_y1, offset_type offset_x2, offset_type offset_y2, scalar_type threshold)
    : offset_x1_(offset_x1), offset_y1_(offset_y1), offset_x2_(offset_x2), offset_y2_(offset_y2), threshold_(threshold) {}

    Direction evaluate(const ImageSample<TPixel>& sample) const
    {
        TPixel pixel_difference = compute_pixel_difference(sample);
        return evaluate(pixel_difference);
    }

    Direction evaluate(TPixel value) const
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

private:
    scalar_type compute_pixel_difference(const ImageSample<TPixel>& sample) const {
        TPixel pixel1_value = compute_pixel_value(sample, offset_x1_, offset_y1_);
        TPixel pixel2_value = compute_pixel_value(sample, offset_x2_, offset_y2_);
        return pixel1_value - pixel2_value;
    }
    
    scalar_type compute_pixel_value(const ImageSample<TPixel>& sample, offset_type offset_x, offset_type offset_y) const {
        const Image<TPixel>& image = sample.get_image();
        offset_type x = sample.get_x();
        offset_type y = sample.get_y();
        TPixel pixel_value;
        if (x + offset_x < 0 || x + offset_x >= image.width() || y + offset_y < 0 || y + offset_y >= image.height())
            pixel_value = 0;
        else
            pixel_value = image.get_data_matrix()(x + offset_x, y + offset_y);
        return pixel_value;
    }
    
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive & offset_x1_;
        archive & offset_y1_;
        archive & offset_x2_;
        archive & offset_y2_;
        archive & threshold_;
    }
#endif
    
    friend class cereal::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("offset_x1", offset_x1_));
        archive(cereal::make_nvp("offset_y1", offset_y1_));
        archive(cereal::make_nvp("offset_x2", offset_x2_));
        archive(cereal::make_nvp("offset_y2", offset_y2_));
        archive(cereal::make_nvp("threshold", threshold_));
    }

    offset_type offset_x1_;
    offset_type offset_y1_;
    offset_type offset_x2_;
    offset_type offset_y2_;
    scalar_type threshold_;
};

template <typename TStatisticsFactory, typename TSampleIterator, typename TRandomEngine = std::mt19937_64, typename TPixel = pixel_type>
class ImageWeakLearner : public WeakLearner<ImageSplitPoint<TPixel>, TStatisticsFactory, TSampleIterator, TRandomEngine>
{
    using BaseT = WeakLearner<ImageSplitPoint<TPixel>, TStatisticsFactory, TSampleIterator, TRandomEngine>;

    const ImageWeakLearnerParameters parameters_;

public:
    using PixelT = TPixel;
    using ParametersT = ImageWeakLearnerParameters;
    using StatisticsT = typename BaseT::StatisticsT;
    using SplitPointT = ImageSplitPoint<TPixel>;

    ImageWeakLearner(const ParametersT& parameters, const TStatisticsFactory& statistics_factory)
    : BaseT(statistics_factory), parameters_(parameters)
    {}

    ~ImageWeakLearner() {}

    virtual std::vector<SplitPointT> sample_split_points(TSampleIterator first_sample, TSampleIterator last_sample, TRandomEngine& rnd_engine) const
    {
        std::vector<SplitPointT> split_points;
        // TODO: Seed with parameter value
        // TOOD: Image width?

        // TODO: Fix discrete offset distributions
        offset_type offset_x_range_low = parameters_.feature_offset_x_range_low;
        offset_type offset_x_range_high = parameters_.feature_offset_x_range_high;
        std::vector<offset_type> offsets_x;
        for (offset_type offset_x=offset_x_range_low; offset_x <= offset_x_range_high; offset_x++) {
            offsets_x.push_back(-offset_x);
            offsets_x.push_back(+offset_x);
        }
        std::uniform_int_distribution<offset_type> offset_x_distribution(0, offsets_x.size() - 1);

        offset_type offset_y_range_low = parameters_.feature_offset_y_range_low;
        offset_type offset_y_range_high = parameters_.feature_offset_y_range_high;
        std::vector<offset_type> offsets_y;
        for (offset_type offset_y=offset_y_range_low; offset_y <= offset_y_range_high; offset_y++) {
            offsets_y.push_back(-offset_y);
            offsets_y.push_back(+offset_y);
        }
        std::uniform_int_distribution<offset_type> offset_y_distribution(0, offsets_y.size() - 1);

        scalar_type threshold_range_low = parameters_.threshold_range_low;
        scalar_type threshold_range_high = parameters_.threshold_range_high;
        std::uniform_real_distribution<scalar_type> threshold_distribution(threshold_range_low, threshold_range_high);

        for (size_type i_f=0; i_f < parameters_.num_of_features; i_f++)
        {
            offset_type offset_x1 = offsets_x[offset_x_distribution(rnd_engine)];
            offset_type offset_y1 = offsets_y[offset_y_distribution(rnd_engine)];
            offset_type offset_x2 = offsets_x[offset_x_distribution(rnd_engine)];
            offset_type offset_y2 = offsets_y[offset_y_distribution(rnd_engine)];
            for (size_type i_t=0; i_t < parameters_.num_of_thresholds; i_t++)
            {
                scalar_type threshold = threshold_distribution(rnd_engine);
                SplitPointT split_point(offset_x1, offset_y1, offset_x2, offset_y2, threshold);
                split_points.push_back(split_point);
            }
        }
        return split_points;
    }

    // TODO: Put SplitPoints into own datastructure to allow computing a feature value once and evaluating it on all thresholds
    virtual SplitStatistics<StatisticsT> compute_split_statistics(TSampleIterator first_sample, TSampleIterator last_sample, const std::vector<SplitPointT>& split_points) const
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

}
