//
//  common.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 19/04/16.
//
//

#pragma once

#include <memory>

#include "image_weak_learner.h"
#include "evaluation_utils.h"

namespace ait {

	using CommonPixelT = pixel_type;
	using CommonLabelT = label_type;
	using CommonImageProviderT = ImageProvider<CommonPixelT>;
	using CommonImageProviderPtrT = std::shared_ptr<CommonImageProviderT>;
	using CommonImageT = Image<CommonPixelT>;
	using CommonImagePtrT = std::shared_ptr<CommonImageT>;

	std::shared_ptr<CommonImageProviderT> get_image_provider_from_image_list(const std::string& image_list_file);

#if WITH_MATLAB
	std::shared_ptr<CommonImageProviderT> get_image_provider_from_matlab_file(const std::string& data_mat_file);
#endif

	void print_image_size(const ait::CommonImageProviderPtrT& image_provider_ptr);

	bool validate_data_ranges(const ait::CommonImageProviderPtrT& image_provider_ptr, int num_of_classes, const CommonLabelT& background_label);

	int compute_num_of_classes(const ait::CommonImageProviderPtrT& image_provider_ptr);

	template <typename TSampleProviderPtr, typename TRandomEngine>
	void load_samples_from_all_images(const TSampleProviderPtr& sample_provider_ptr, TRandomEngine& rnd_engine)
	{
        sample_provider_ptr->clear_samples();
        for (int i = 0; i < sample_provider_ptr->get_num_of_images(); ++i) {
        	sample_provider_ptr->load_samples_from_image(i, rnd_engine);
        }
	}

	template <typename TForest, typename TSampleProviderPtr>
	void print_sample_counts(const TForest& forest, const TSampleProviderPtr& sample_provider_ptr, int num_of_classes)
	{
        auto samples_start = sample_provider_ptr->get_samples_begin();
        auto samples_end = sample_provider_ptr->get_samples_end();
        std::vector<ait::size_type> sample_counts(num_of_classes, 0);
        for (auto sample_it = samples_start; sample_it != samples_end; sample_it++) {
            ++sample_counts[sample_it->get_label()];
        }
        auto logger = ait::log_info(true);
        logger << "Sample counts>> ";
        for (int c = 0; c < num_of_classes; ++c) {
            if (c > 0) {
                logger << ", ";
            }
            logger << "class " << c << ": " << sample_counts[c];
        }
        logger.close();
	}

	template <typename TForest, typename TSampleProviderPtr>
	void print_match_counts(const TForest& forest, const TSampleProviderPtr& sample_provider_ptr)
	{
        auto samples_start = sample_provider_ptr->get_samples_begin();
        auto samples_end = sample_provider_ptr->get_samples_end();
        // For each tree extract leaf node indices for each sample.
        std::vector<std::vector<ait::size_type>> forest_leaf_indices = forest.evaluate(samples_start, samples_end);
        // Compute number of prediction matches based on a majority vote among the forest.
        int match = 0;
        int no_match = 0;
        for (auto tree_it = forest.cbegin(); tree_it != forest.cend(); ++tree_it) {
            for (auto sample_it = samples_start; sample_it != samples_end; sample_it++) {
                const auto &node_it = tree_it->cbegin() + (forest_leaf_indices[tree_it - forest.cbegin()][sample_it - samples_start]);
                const auto &statistics = node_it->get_statistics();
                auto max_it = std::max_element(statistics.get_histogram().cbegin(), statistics.get_histogram().cend());
                auto label = max_it - statistics.get_histogram().cbegin();
                if (label == sample_it->get_label()) {
                    match++;
                } else {
                    no_match++;
                }
            }
        }
        ait::log_info() << "Match: " << match << ", no match: " << no_match;
	}

	template <typename TForest, typename TSampleProviderPtr>
	void print_per_pixel_confusion_matrix(const TForest& forest, const TSampleProviderPtr& sample_provider_ptr, int num_of_classes)
	{
        auto samples_start = sample_provider_ptr->get_samples_begin();
        auto samples_end = sample_provider_ptr->get_samples_end();
        // Compute confusion matrix.
        auto forest_utils = ait::make_forest_utils(forest);
        auto confusion_matrix = forest_utils.compute_confusion_matrix(samples_start, samples_end);
        ait::log_info() << "Confusion matrix:" << std::endl << confusion_matrix;
        auto norm_confusion_matrix = ait::EvaluationUtils::normalize_confusion_matrix(confusion_matrix);
        ait::log_info() << "Normalized confusion matrix:" << std::endl << norm_confusion_matrix;
        ait::log_info() << "Diagonal of normalized confusion matrix:" << std::endl << norm_confusion_matrix.diagonal();
        ait::log_info() << "Mean diagonal of normalized confusion matrix:" << std::endl << norm_confusion_matrix.diagonal().mean();
	}

	template <typename TForest, typename TSampleProviderPtr, typename TRandomEngine>
	void print_per_frame_confusion_matrix(const TForest& forest, const TSampleProviderPtr& full_sample_provider_ptr, TRandomEngine& rnd_engine, int num_of_classes)
	{
        // Computing per-frame confusion matrix
        auto forest_utils = ait::make_forest_utils(forest);
        using ConfusionMatrixType = typename decltype(forest_utils)::MatrixType;
        ConfusionMatrixType per_frame_confusion_matrix(num_of_classes, num_of_classes);
        per_frame_confusion_matrix.setZero();
        for (int i = 0; i < full_sample_provider_ptr->get_num_of_images(); ++i) {
        	full_sample_provider_ptr->clear_samples();
        	full_sample_provider_ptr->load_samples_from_image(i, rnd_engine);
            auto samples_start = full_sample_provider_ptr->get_samples_begin();
            auto samples_end = full_sample_provider_ptr->get_samples_end();
            forest_utils.update_confusion_matrix(per_frame_confusion_matrix, samples_start, samples_end);
        }
        ait::log_info() << "Per-frame confusion matrix:" << std::endl << per_frame_confusion_matrix;
        ConfusionMatrixType per_frame_norm_confusion_matrix = ait::EvaluationUtils::normalize_confusion_matrix(per_frame_confusion_matrix);
        ait::log_info() << "Normalized per-frame confusion matrix:" << std::endl << per_frame_norm_confusion_matrix;
        ait::log_info() << "Diagonal of normalized per-frame confusion matrix:" << std::endl << per_frame_norm_confusion_matrix.diagonal();
        ait::log_info() << "Mean of diagonal of normalized per-frame confusion matrix:" << std::endl << per_frame_norm_confusion_matrix.diagonal().mean();
	}

}
