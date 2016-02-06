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
#include <algorithm>
#if AIT_MULTI_THREADING
#include <thread>
#endif

#ifdef SERIALIZE_WITH_BOOST
#include <boost/serialization/vector.hpp>
#include "serialization_utils.h"
#endif
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>
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

class ImageParameters
{
public:
#if AIT_TESTING
//    double samples_per_image_fraction = 0.015;
//    double bagging_fraction = 0.1;
    double samples_per_image_fraction = 0.2;
    double bagging_fraction = 1.0;
#else
    double samples_per_image_fraction = 0.2;
    double bagging_fraction = 1.0;
#endif
    label_type background_label = -1;
};

class ImageWeakLearnerParameters : public ImageParameters
{
public:
#if AIT_TESTING
//    int num_of_thresholds = 10;
//    int num_of_features = 10;
    int num_of_thresholds = 100;
    int num_of_features = 100;
#else
    int num_of_thresholds = 100;
    int num_of_features = 100;
#endif
    offset_type feature_offset_x_range_low = 3;
    offset_type feature_offset_x_range_high = 15;
    offset_type feature_offset_y_range_low = 3;
    offset_type feature_offset_y_range_high = 15;
    scalar_type threshold_range_low = -300.0;
    scalar_type threshold_range_high = +300;
    bool adaptive_threshold_range = true;
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
    explicit Image()
    {
    }

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
        assert(width == label_image.width());
        assert(height == label_image.height());
        assert(depth == label_image.depth());
        assert(spectrum == label_image.spectrum());
        assert(depth == 1);
        assert(spectrum == 1);
        if (width != label_image.width() || height != label_image.height())
        {
            throw std::runtime_error("Data and label images need to have the same size");
        }
        if (depth != 1 || depth != label_image.depth())
        {
            throw std::runtime_error("Images need to have a depth of 1 (CImg depth)");
        }
        if (spectrum != 1 || spectrum != label_image.spectrum())
        {
            throw std::runtime_error("Images need to have a spectrum of 1 (CImg spectrum)");
        }
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

    explicit ImageSample(const Image<TPixel>* image_ptr, offset_type x, offset_type y)
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
class ImageSampleProvider
{
public:
    using SampleT = ImageSample<TPixel>;
    using SampleIteratorT = typename std::vector<SampleT>::iterator;
    using ConstSampleIteratorT = typename std::vector<SampleT>::const_iterator;
    using SampleBagBatchT = std::vector<size_type>;
    using ImageT = Image<TPixel>;
    using ParametersT = ImageParameters;

    explicit ImageSampleProvider(const std::vector<std::tuple<std::string, std::string>>& image_list, const ImageParameters& parameters)
    : image_list_(image_list), parameters_(parameters)
    {
        assert(image_list.size() > 0);
        ImageT image = load_image(0);
        image_width_ = image.width();
        image_height_ = image.height();
    }

    std::vector<SampleBagBatchT> compute_sample_bag_batches(size_type num_of_batches, TRandomEngine& rnd_engine) const
	{
        int num_of_images_per_bag = std::round(parameters_.bagging_fraction * image_list_.size());
        std::vector<size_type> image_indices(num_of_images_per_bag);
        std::uniform_int_distribution<> image_dist(0, image_list_.size() - 1);
        for (int i = 0; i < num_of_images_per_bag; i++)
        {
            int image_index = image_dist(rnd_engine);
			image_indices[i] = image_index;
        }
        std::sort(image_indices.begin(), image_indices.end());
        std::vector<SampleBagBatchT> split_sample_bag(num_of_batches);
        for (int i = 0; i < num_of_batches; i++)
        {
            size_type index_start = compute_batch_start_index(i, num_of_batches, num_of_images_per_bag);
            size_type index_end = compute_batch_start_index(i + 1, num_of_batches, num_of_images_per_bag);
            auto bag_it_start = image_indices.cbegin() + index_start;
            auto bag_it_end = image_indices.cbegin() + index_end;
            SampleBagBatchT image_indices_batch(bag_it_start, bag_it_end);
        	split_sample_bag[i] = std::move(image_indices_batch);
        }
        return split_sample_bag;
    }

    void load_sample_batch(const SampleBagBatchT& split_sample_item, TRandomEngine& rnd_engine)
    {
        log_info(false) << "Loading split sample bag ...";
        clear_samples();
        std::map<size_type, ImageT> old_image_map(std::move(image_map_));
        image_map_.clear();
        for (auto it = split_sample_item.cbegin(); it != split_sample_item.cend(); ++it)
        {
            size_type image_index = *it;
            ensure_image_is_loaded(image_index, old_image_map);
			load_samples_from_image(image_index, rnd_engine);
        }
        log_info(true) << "Done";
    }

    void load_samples_from_image(size_type image_index, TRandomEngine& rnd_engine)
    {
    	ensure_image_is_loaded(image_index);
		const ImageT* image_ptr = &image_map_.at(image_index);
		if (parameters_.samples_per_image_fraction < 1.0)
		{
			int num_of_samples_per_image = std::round(parameters_.samples_per_image_fraction * image_width_ * image_height_);
            std::vector<SampleT> non_background_samples;
            for (int x = 0; x < image_width_; ++x)
            {
                for (int y = 0; y < image_height_; ++y)
                {
                    SampleT sample(image_ptr, x, y);
                    if (sample.get_label() != parameters_.background_label)
                    {
                        non_background_samples.push_back(std::move(sample));
                    }
                }
            }
            int num_of_samples = std::min(num_of_samples_per_image, static_cast<int>(non_background_samples.size()));
            for (int i = 0; i < num_of_samples; ++i)
            {
                std::uniform_int_distribution<> index_dist(0, non_background_samples.size() - 1 - i);
                int index = index_dist(rnd_engine);
                std::swap(non_background_samples[index], non_background_samples.back());
                samples_.push_back(std::move(non_background_samples.back()));			}
		}
		else
		{
			for (int x = 0; x < image_width_; ++x)
			{
				for (int y = 0; y < image_height_; ++y)
				{
					SampleT sample(image_ptr, x, y);
                    label_type label = sample.get_label();
					if (label != parameters_.background_label)
					{
						samples_.push_back(std::move(sample));
					}
				}
			}
		}
    }

    void load_sample_bag(TRandomEngine& rnd_engine)
    {
    	std::vector<SampleBagBatchT> sample_bag_batches = compute_sample_bag_batches(1, rnd_engine);
		load_sample_batch(sample_bag_batches[0], rnd_engine);
    }

    void clear_image_cache()
    {
    	image_map_.clear();
    }

    void clear_samples()
    {
        samples_.clear();
    }
    
    SampleIteratorT get_samples_begin()
    {
        return samples_.begin();
    }
    
    SampleIteratorT get_samples_end()
    {
        return samples_.end();
    }

    ConstSampleIteratorT get_samples_cbegin() const
    {
        return samples_.cbegin();
    }

    ConstSampleIteratorT get_samples_cend() const
    {
        return samples_.cend();
    }

private:
    void ensure_image_is_loaded(size_type image_index)
    {
		typename std::map<size_type, ImageT>::const_iterator image_it = image_map_.find(image_index);
		if (image_it == image_map_.cend())
		{
			// Load image into memory.
			ImageT image = load_image(image_index);
			image_map_[image_index] = std::move(image);
		}
    }

    void ensure_image_is_loaded(size_type image_index, const std::map<size_type, ImageT>& old_image_map)
    {
		typename std::map<size_type, ImageT>::const_iterator image_it = image_map_.find(image_index);
		if (image_it == image_map_.cend())
		{
			// Image is not yet in the image map.
			image_it = old_image_map.find(image_index);
			if (image_it != old_image_map.cend())
			{
				// Image was found in the previously used image map (cached).
				image_map_[image_index] = std::move(image_it->second);
			}
			else
			{
				// Load image into memory.
				ImageT image = load_image(image_index);
				image_map_[image_index] = std::move(image);
			}
		}
    }

    ImageT load_image(size_type image_index) const
    {
        const std::string& data_path = std::get<0>(image_list_[image_index]);
        const std::string& label_path = std::get<1>(image_list_[image_index]);
        ImageT image = ImageT::load_from_files(data_path, label_path);
        return image;
    }

    size_type compute_batch_start_index(size_type batch_index, size_type num_of_batches, size_type batch_size) const
    {
        return static_cast<size_type>(batch_index * batch_size / static_cast<double>(num_of_batches));
    }

    size_type image_width_;
    size_type image_height_;
    const std::vector<std::tuple<std::string, std::string>> image_list_;
    const ImageParameters parameters_;
    std::map<size_type, ImageT> image_map_;
    std::vector<SampleT> samples_;
};

struct ImageFeature
{
    explicit ImageFeature()
    : offset_x1(0), offset_y1(0), offset_x2(0), offset_y2(0)
    {}

    explicit ImageFeature(offset_type offset_x1, offset_type offset_y1, offset_type offset_x2, offset_type offset_y2)
    : offset_x1(offset_x1), offset_y1(offset_y1), offset_x2(offset_x2), offset_y2(offset_y2)
    {}

    template <typename TPixel>
    scalar_type compute_pixel_difference(const ImageSample<TPixel>& sample) const {
        TPixel pixel1_value = compute_pixel_value(sample, offset_x1, offset_y1);
        TPixel pixel2_value = compute_pixel_value(sample, offset_x2, offset_y2);
        return pixel1_value - pixel2_value;
    }
    
    template <typename TPixel>
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

    // TODO: remove
//    bool operator<(const ImageFeature& other) const
//    {
//        if (offset_x1 < other.offset_x1)
//        {
//            return true;
//        }
//        else if (offset_x1 == other.offset_x1)
//        {
//            if (offset_y1 < other.offset_y1)
//            {
//                return true;
//            }
//            else if (offset_y1 == other.offset_y1)
//            {
//                if (offset_x2 < other.offset_x2)
//                {
//                    return true;
//                }
//                else if (offset_x2 == other.offset_x2)
//                {
//                    if (offset_y2 < other.offset_y2)
//                    {
//                        return true;
//                    }
//                }
//            }
//        }
//        return false;
//    }

    offset_type offset_x1;
    offset_type offset_y1;
    offset_type offset_x2;
    offset_type offset_y2;
    
private:
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive & offset_x1;
        archive & offset_y1;
        archive & offset_x2;
        archive & offset_y2;
    }
#endif

    friend class cereal::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("offset_x1", offset_x1));
        archive(cereal::make_nvp("offset_y1", offset_y1));
        archive(cereal::make_nvp("offset_x2", offset_x2));
        archive(cereal::make_nvp("offset_y2", offset_y2));
    }
};

struct ImageThreshold
{
    explicit ImageThreshold()
    : threshold(0)
    {}

    explicit ImageThreshold(scalar_type threshold)
    : threshold(threshold)
    {}

    bool left_direction(scalar_type value) const
    {
        return value < threshold;
    }

    scalar_type threshold;

private:
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive & threshold;
    }
#endif
    
    friend class cereal::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("threshold", threshold));
    }
};

template <typename TPixel = pixel_type>
class ImageSplitPoint {
public:
    using PixelT = TPixel;

    explicit ImageSplitPoint()
    : offset_x1_(0), offset_y1_(0), offset_x2_(0), offset_y2_(0), threshold_(0)
    {}

    explicit ImageSplitPoint(offset_type offset_x1, offset_type offset_y1, offset_type offset_x2, offset_type offset_y2, scalar_type threshold)
    : offset_x1_(offset_x1), offset_y1_(offset_y1), offset_x2_(offset_x2), offset_y2_(offset_y2), threshold_(threshold)
    {}

    explicit ImageSplitPoint(const ImageFeature& feature, const ImageThreshold& threshold)
    : offset_x1_(feature.offset_x1), offset_y1_(feature.offset_y1), offset_x2_(feature.offset_x2), offset_y2_(feature.offset_y2), threshold_(threshold.threshold)
    {}

    Direction evaluate(const ImageSample<PixelT>& sample) const
    {
        PixelT pixel_difference = compute_pixel_difference(sample);
        return evaluate(pixel_difference);
    }

    Direction evaluate(PixelT value) const
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
    scalar_type compute_pixel_difference(const ImageSample<PixelT>& sample) const {
        PixelT pixel1_value = compute_pixel_value(sample, offset_x1_, offset_y1_);
        PixelT pixel2_value = compute_pixel_value(sample, offset_x2_, offset_y2_);
        return pixel1_value - pixel2_value;
    }
    
    scalar_type compute_pixel_value(const ImageSample<PixelT>& sample, offset_type offset_x, offset_type offset_y) const {
        const Image<PixelT>& image = sample.get_image();
        offset_type x = sample.get_x();
        offset_type y = sample.get_y();
        PixelT pixel_value;
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

template <typename TPixel = pixel_type>
class ImageSplitPointCandidates
{
public:
    using SplitPointT = ImageSplitPoint<pixel_type>;

    using iterator = typename std::vector<std::tuple<ImageFeature, std::vector<ImageThreshold>>>::iterator;
    using const_iterator = typename std::vector<std::tuple<ImageFeature, std::vector<ImageThreshold>>>::const_iterator;

    explicit ImageSplitPointCandidates()
    : total_size_(0)
    {}

    void add_feature_and_thresholds(const ImageFeature& feature, const std::vector<ImageThreshold>& thresholds)
    {
        candidates_.push_back(std::make_tuple(feature, thresholds));
        total_size_ += thresholds.size();
    }

    SplitPointT get_split_point(size_type index) const
    {
        assert(index < total_size());
        size_type i = 0;
        for (const_iterator it = cbegin(); it != cend(); ++it)
        {
            const ImageFeature& feature = std::get<0>(*it);
            const std::vector<ImageThreshold>& thresholds = std::get<1>(*it);
            for (auto threshold_it = thresholds.cbegin(); threshold_it != thresholds.cend(); ++threshold_it)
            {
                if (i == index)
                {
                    return ImageSplitPoint<TPixel>(feature, *threshold_it);
                }
                ++i;
            }
        }
        throw std::invalid_argument("Could not find split point with the corresponding index.");
    }
    
    size_type size() const
    {
        return candidates_.size();
    }
    

    size_type total_size() const
    {
        return total_size_;
    }

    iterator begin()
    {
        return candidates_.begin();
    }
    
    iterator end()
    {
        return candidates_.end();
    }
    
    const_iterator cbegin() const
    {
        return candidates_.cbegin();
    }
    
    const_iterator cend() const
    {
        return candidates_.cend();
    }

private:
#ifdef SERIALIZE_WITH_BOOST
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename enable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive & candidates_;
        archive & total_size_;
    }
#endif
    
    friend class cereal::access;
    
    template <typename Archive>
    void serialize(Archive& archive, const unsigned int version, typename disable_if_boost_archive<Archive>::type* = nullptr)
    {
        archive(cereal::make_nvp("candidates", candidates_));
        archive(cereal::make_nvp("total_size", total_size_));
    }
    
    std::vector<std::tuple<ImageFeature, std::vector<ImageThreshold>>> candidates_;
    size_type total_size_;
};

template <typename TStatisticsFactory, typename TSampleIterator, typename TRandomEngine = std::mt19937_64, typename TPixel = pixel_type>
class ImageWeakLearner : public WeakLearner<ImageSplitPointCandidates<TPixel>, TStatisticsFactory, TSampleIterator, TRandomEngine>
{
    using BaseT = WeakLearner<ImageSplitPointCandidates<TPixel>, TStatisticsFactory, TSampleIterator, TRandomEngine>;

    const ImageWeakLearnerParameters parameters_;

public:
    using PixelT = TPixel;
    using ParametersT = ImageWeakLearnerParameters;
    using StatisticsT = typename BaseT::StatisticsT;
    using SplitPointT = ImageSplitPoint<TPixel>;
    using SplitPointCandidatesT = typename BaseT::SplitPointCandidatesT;

    ImageWeakLearner(const ParametersT& parameters, const TStatisticsFactory& statistics_factory)
    : BaseT(statistics_factory), parameters_(parameters)
    {}

    ~ImageWeakLearner() {}

    void compute_adaptive_threshold_range(TSampleIterator first_sample, TSampleIterator last_sample, const ImageFeature& feature, scalar_type* threshold_range_low, scalar_type* threshold_range_high) const
    {
        scalar_type min_value = std::numeric_limits<scalar_type>::max();
        scalar_type max_value = std::numeric_limits<scalar_type>::min();
        for (TSampleIterator sample_it = first_sample; sample_it != last_sample; sample_it++)
        {
            scalar_type value = feature.compute_pixel_difference(*sample_it);
            if (value < min_value)
            {
                min_value = value;
            }
            if (value > max_value)
            {
                max_value = value;
            }
        }
        if (min_value >= max_value)
        {
            min_value = 0;
            max_value = 0;
        }
        *threshold_range_low = min_value;
        *threshold_range_high = max_value;
    }

    virtual SplitPointCandidatesT sample_split_points(TSampleIterator first_sample, TSampleIterator last_sample, TRandomEngine& rnd_engine) const
    {
        SplitPointCandidatesT split_points;

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

        for (size_type i_f=0; i_f < parameters_.num_of_features; i_f++)
        {
            offset_type offset_x1 = offsets_x[offset_x_distribution(rnd_engine)];
            offset_type offset_y1 = offsets_y[offset_y_distribution(rnd_engine)];
            offset_type offset_x2 = offsets_x[offset_x_distribution(rnd_engine)];
            offset_type offset_y2 = offsets_y[offset_y_distribution(rnd_engine)];
            ImageFeature feature(offset_x1, offset_y1, offset_x2, offset_y2);
            // Optional: Compute adaptive threshold range
            if (parameters_.adaptive_threshold_range)
            {
                compute_adaptive_threshold_range(first_sample, last_sample, feature, &threshold_range_low, &threshold_range_high);
            }
            std::uniform_real_distribution<scalar_type> threshold_distribution(threshold_range_low, threshold_range_high);
            std::vector<ImageThreshold> thresholds;
            for (size_type i_t=0; i_t < parameters_.num_of_thresholds; i_t++)
            {
                scalar_type threshold = threshold_distribution(rnd_engine);
                thresholds.push_back(ImageThreshold(threshold));
            }
            split_points.add_feature_and_thresholds(feature, thresholds);
        }
        return split_points;
    }

    void _compute_split_statistics(TSampleIterator first_sample, TSampleIterator last_sample, SplitStatistics<StatisticsT>& split_statistics, size_type statistics_index_offset, const ImageFeature& feature, const std::vector<ImageThreshold>& thresholds) const
    {
        for (TSampleIterator sample_it = first_sample; sample_it != last_sample; sample_it++)
        {
            scalar_type value = feature.compute_pixel_difference(*sample_it);
            size_type statistics_index = statistics_index_offset;
            for (auto threshold_it = thresholds.cbegin(); threshold_it != thresholds.cend(); ++threshold_it)
            {
                if (threshold_it->left_direction(value))
                {
                    split_statistics.get_left_statistics(statistics_index).lazy_accumulate(*sample_it);
                }
                else
                {
                    split_statistics.get_right_statistics(statistics_index).lazy_accumulate(*sample_it);
                }
                ++statistics_index;
            }
        }

        for (auto threshold_it = thresholds.cbegin(); threshold_it != thresholds.cend(); ++threshold_it)
        {
            size_type statistics_index = statistics_index_offset;
            split_statistics.get_left_statistics(statistics_index).finish_lazy_accumulation();
            split_statistics.get_right_statistics(statistics_index).finish_lazy_accumulation();
            ++statistics_index;
        }
    }

    virtual SplitStatistics<StatisticsT> compute_split_statistics(TSampleIterator first_sample, TSampleIterator last_sample, const SplitPointCandidatesT& split_points) const
    {
        // we create statistics for all features and thresholds here so that we can easily parallelize the loop below
        SplitStatistics<StatisticsT> split_statistics(split_points.size(), this->statistics_factory_);
        for (typename SplitPointCandidatesT::const_iterator it = split_points.cbegin(); it != split_points.cend(); ++it)
        {
            const ImageFeature& feature = std::get<0>(*it);
            const std::vector<ImageThreshold>& thresholds = std::get<1>(*it);
            
            size_type statistics_index_offset = (it - split_points.cbegin()) * thresholds.size();
            _compute_split_statistics(first_sample, last_sample, split_statistics, statistics_index_offset, feature, thresholds);
        }

        return split_statistics;
    }
    
#if AIT_MULTI_THREADING
    virtual SplitStatistics<StatisticsT> compute_split_statistics_parallel(TSampleIterator first_sample, TSampleIterator last_sample, const SplitPointCandidatesT& split_points, int num_of_threads) const
    {
        if (num_of_threads <= 0)
        {
            num_of_threads = std::thread::hardware_concurrency();
        }
        std::vector<std::thread> threads;

        // we create statistics for all features and thresholds here so that we can easily parallelize the loop below
        SplitStatistics<StatisticsT> split_statistics(split_points.total_size(), this->statistics_factory_);

        for (int thread_index = 0; thread_index < num_of_threads; ++thread_index)
        {
            size_type split_point_index_begin = std::floor(thread_index * split_points.size() / static_cast<double>(num_of_threads));
            size_type split_point_index_end = std::floor((thread_index + 1) * split_points.size() / static_cast<double>(num_of_threads));
            if (thread_index == num_of_threads - 1)
            {
                assert(split_point_index_end == split_points.size());
            }
            auto thread_lambda = [this, split_point_index_begin, split_point_index_end, &split_statistics, &split_points, &first_sample, &last_sample]()
            {
                for (size_type split_point_index = split_point_index_begin; split_point_index < split_point_index_end; ++split_point_index)
                {
                    typename SplitPointCandidatesT::const_iterator it = split_points.cbegin() + split_point_index;
                    const ImageFeature& feature = std::get<0>(*it);
                    const std::vector<ImageThreshold>& thresholds = std::get<1>(*it);

                    size_type statistics_index_offset = (it - split_points.cbegin()) * thresholds.size();
                    _compute_split_statistics(first_sample, last_sample, split_statistics, statistics_index_offset, feature, thresholds);
                }
            };
            std::thread thread(thread_lambda);
            threads.push_back(std::move(thread));
        }

        for (std::thread& thread : threads)
        {
            thread.join();
        }
        
        return split_statistics;
    }
#endif

};

}
