//
//  common.cpp
//  DistRandomForest
//
//  Created by Benjamin Hepp on 19/04/16.
//
//

#include "common.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include <boost/filesystem/path.hpp>

#include "image_weak_learner.h"
#include "logger.h"
#include "csv_utils.h"

#if WITH_MATLAB
#include "matlab_file_io.h"
#endif

std::shared_ptr<ait::CommonImageProviderT> ait::get_image_provider_from_image_list(const std::string& image_list_file)
{
    // Read image file list
    ait::log_info(false) << "Reading image list ... " << std::flush;
    std::vector<std::tuple<std::string, std::string>> image_list;
    std::ifstream ifile(image_list_file);
    if (!ifile.good()) {
        throw std::runtime_error("Unable to open image list file");
    }
    ait::CSVReader<std::string> csv_reader(ifile);
    for (auto it = csv_reader.begin(); it != csv_reader.end(); ++it) {
        if (it->size() != 2) {
            throw std::runtime_error("Image list file must contain two columns with the data and label filenames.");
        }
        const std::string& data_filename = (*it)[0];
        const std::string& label_filename = (*it)[1];

        boost::filesystem::path data_path = boost::filesystem::path(data_filename);
        boost::filesystem::path label_path = boost::filesystem::path(label_filename);
        if (!data_path.is_absolute()) {
            data_path = boost::filesystem::path(image_list_file).parent_path();
            data_path /= data_filename;
        }
        if (!label_path.is_absolute()) {
            label_path = boost::filesystem::path(image_list_file).parent_path();
            label_path /= label_filename;
        }

        image_list.push_back(std::make_tuple(data_path.string(), label_path.string()));
    }
    ait::log_info(false) << " Done." << std::endl;

	return std::make_shared<ait::FileImageProvider<ait::CommonPixelT>>(image_list);
}

#if WITH_MATLAB
std::shared_ptr<ait::CommonImageProviderT> ait::get_image_provider_from_matlab_file(const std::string& data_mat_file)
{
    ait::log_info(false) << "Reading images from Matlab file... " << std::flush;
    auto images = ait::load_images_from_matlab_file(data_mat_file, "data", "labels");
    ait::log_info(false) << " Done." << std::endl;

	return std::make_shared<ait::MemoryImageProvider<ait::CommonPixelT>>(std::move(images));
}
#endif

void ait::print_image_size(const ait::CommonImageProviderPtrT& image_provider_ptr)
{
	// Print size of images
	ait::size_type image_height = image_provider_ptr->get_image(0)->get_data_matrix().rows();
	ait::size_type image_width = image_provider_ptr->get_image(0)->get_data_matrix().cols();
	ait::log_info(false) << "Image size " << image_width << " x " << image_height << std::endl;
}

bool ait::validate_data_ranges(const ait::CommonImageProviderPtrT& image_provider_ptr, int num_of_classes, const ait::CommonLabelT& background_label)
{
	// Compute value range of data and labels
	ait::log_info() << "Computing label and value range of data ...";
	ait::pixel_type max_value = std::numeric_limits<ait::pixel_type>::min();
	ait::pixel_type min_value = std::numeric_limits<ait::pixel_type>::max();
	ait::label_type max_label = std::numeric_limits<ait::label_type>::min();
	ait::label_type min_label = std::numeric_limits<ait::label_type>::max();
	ait::label_type fg_max_label = std::numeric_limits<ait::label_type>::min();
	ait::label_type fg_min_label = std::numeric_limits<ait::label_type>::max();
	for (auto i = 0; i < image_provider_ptr->get_num_of_images(); i++) {
		const auto image_ptr = image_provider_ptr->get_image(i);
		const auto&  data_matrix = image_ptr->get_data_matrix();
		ait::pixel_type local_min_value = data_matrix.minCoeff();
		min_value = std::min(local_min_value, min_value);
		ait::pixel_type local_max_value = data_matrix.maxCoeff();
		max_value = std::max(local_max_value, max_value);
		const auto& label_matrix = image_ptr->get_label_matrix();
		ait::label_type local_min_label = label_matrix.minCoeff();
		min_label = std::min(local_min_label, min_label);
		ait::label_type local_max_label = label_matrix.maxCoeff();
		max_label = std::max(local_max_label, max_label);
        for (int x = 0; x < label_matrix.rows(); ++x) {
            for (int y = 0; y < label_matrix.cols(); ++y) {
                auto label = label_matrix(x, y);
                if (label != background_label) {
                	fg_min_label = std::min(label, fg_min_label);
                	fg_max_label = std::max(label, fg_max_label);
                }
            }
        }
	}
	ait::log_info() << " Value range [" << min_value << ", " << max_value << "].";
	ait::log_info() << " Label range [" << min_label << ", " << max_label << "].";
	ait::log_info() << " Foreground label range [" << fg_min_label << ", " << fg_max_label << "].";
	if (fg_min_label < 0 || fg_max_label >= num_of_classes) {
		ait::log_error() << "Foreground label ranges do not match number of classes: " << num_of_classes;
		return false;
	}
	return true;
}

int ait::compute_num_of_classes(const ait::CommonImageProviderPtrT& image_provider_ptr)
{
    ait::label_type max_label = 0;
    for (auto i = 0; i < image_provider_ptr->get_num_of_images(); i++) {
    	const auto image_ptr = image_provider_ptr->get_image(i);
		const auto& label_matrix = image_ptr->get_label_matrix();
        ait::label_type local_max_label = label_matrix.maxCoeff();
        max_label = std::max(local_max_label, max_label);
    }
    int num_of_classes = static_cast<ait::size_type>(max_label) + 1;
    return num_of_classes;
}
