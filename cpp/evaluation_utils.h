//
//  evaluation_utils.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

namespace ait {

class EvaluationUtils {
public:
	EvaluationUtils() = delete;
    
    template <typename TStatistics, typename TSampleIterator>
    static TStatistics compute_true_statistics(size_type num_of_classes, const TSampleIterator& samples_start, const TSampleIterator& samples_end)
    {
        TStatistics true_statistics(num_of_classes);
        for (auto sample_it = samples_start; sample_it != samples_end; ++sample_it) {
            size_type true_label = sample_it->get_label();
            true_statistics.accumulate(true_label);
        }
        return true_statistics;
    }

	template <typename TMatrix, typename TStatistics>
	static TMatrix compute_confusion_matrix(const std::vector<size_type>& true_labels, const std::vector<TStatistics>& statistics_vector)
	{
		assert(true_labels.size() == statistics_vector.size());
		if (statistics_vector.size() == 0) {
			return TMatrix(0, 0);
		}
		size_type num_of_labels = statistics_vector.front().num_of_bins();
		TMatrix confusion_matrix(num_of_labels, num_of_labels);
		confusion_matrix.setZero();
		for (size_type i = 0; i < true_labels.size(); ++i) {
			size_type true_label = true_labels[i];
			size_type predicted_label = statistics_vector[i].get_max_bin();
			++confusion_matrix(true_label, predicted_label);
		}
		return confusion_matrix;
	}

	template <typename TMatrix>
	static TMatrix& update_confusion_matrix(TMatrix& confusion_matrix, size_type true_label, size_type predicted_label)
	{
		++confusion_matrix(true_label, predicted_label);
		return confusion_matrix;
	}

	template <typename TMatrix, typename TStatistics>
	static TMatrix& update_confusion_matrix(TMatrix& confusion_matrix, size_type true_label, const TStatistics& predicted_statistics)
	{
		size_type predicted_label = predicted_statistics.get_max_bin();
		return update_confusion_matrix(confusion_matrix, true_label, predicted_label);
	}

	template <typename TMatrix, typename TStatistics>
	static TMatrix& update_confusion_matrix(TMatrix& confusion_matrix, const TStatistics& true_statistics, size_type predicted_label)
	{
		size_type true_label = true_statistics.get_max_bin();
		return update_confusion_matrix(confusion_matrix, true_label, predicted_label);
	}

	template <typename TMatrix, typename TStatistics>
	static TMatrix& update_confusion_matrix(TMatrix& confusion_matrix, const TStatistics& true_statistics, const TStatistics& predicted_statistics)
	{
		size_type true_label = true_statistics.get_max_bin();
		size_type predicted_label = predicted_statistics.get_max_bin();
		return update_confusion_matrix(confusion_matrix, true_label, predicted_label);
	}

	template <typename TMatrix>
    static TMatrix normalize_confusion_matrix(const TMatrix& confusion_matrix)
    {
        auto row_sum = confusion_matrix.rowwise().sum();
        TMatrix normalized_confusion_matrix = confusion_matrix;
        for (int col=0; col < confusion_matrix.cols(); col++)
        {
            for (int row=0; row < confusion_matrix.rows(); row++)
            {
                normalized_confusion_matrix(row, col) /= row_sum(row);
            }
        }
        return normalized_confusion_matrix;
    }
};

template <typename TSplitPoint, typename TStatistics, typename TMatrix = Eigen::MatrixXd>
class TreeUtilities
{
	using TreeType = Tree<TSplitPoint, TStatistics>;

	const TreeType& tree_;
    size_type num_of_classes_;

public:
	using MatrixType = TMatrix;

	TreeUtilities(const TreeType& tree)
		: tree_(tree)
    {
        num_of_classes_ = tree_.get_root_iterator()->get_statistics().num_of_bins();
    }

	template <typename TSample>
	const TStatistics& compute_statistics(const TSample& sample) const
	{
		typename TreeType::ConstNodeIterator node_it = tree_.evaluate(sample);
		return node_it->get_statistics();
	}

	template <typename TSampleIterator>
	TStatistics compute_summed_statistics(const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
	{
        TStatistics summed_statistics(num_of_classes_);
        for (TSampleIterator sample_it = samples_start; sample_it != samples_end; ++sample_it) {
            const TStatistics& node_statistics = compute_statistics(*sample_it);
            assert(num_of_classes_ == node_statistics.num_of_bins());
            summed_statistics.lazy_accumulate(node_statistics.get_max_bin());
//            summed_statistics.accumulate(node_statistics);
        }
        summed_statistics.finish_lazy_accumulation();
        return summed_statistics;
	}

	template <typename TSampleIterator>
	TStatistics compute_multiplied_statistics(const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
	{
		TStatistics multiplied_statistics(num_of_classes_);
		for (size_type i = 0; i < num_of_classes_; ++i) {
			multiplied_statistics.get_histogram()[i] = 1;
		}
        for (TSampleIterator sample_it = samples_start; sample_it != samples_end; ++sample_it) {
			const TStatistics& node_statistics = compute_statistics(*sample_it);
			assert(num_of_classes_ == node_statistics.num_of_bins());
			for (size_type i = 0; i < num_of_classes_; ++i) {
				multiplied_statistics.get_histogram()[i] *= node_statistics.get_histogram()[i];
			}
		}
		return multiplied_statistics;
	}

	template <typename TSampleIterator>
	TMatrix compute_confusion_matrix(const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
	{
		TMatrix confusion_matrix(num_of_classes_, num_of_classes_);
        confusion_matrix.setZero();
        for (TSampleIterator sample_it = samples_start; sample_it != samples_end; ++sample_it) {
            update_confusion_matrix(confusion_matrix, *sample_it);
		}
		return confusion_matrix;
	}

	template <typename TTMatrix, typename TSample>
	TTMatrix& update_confusion_matrix(TTMatrix& confusion_matrix, const TSample& sample) const
    {
        size_type true_label = sample.get_label();
		const TStatistics& predicted_statistics = compute_statistics(sample);
        return EvaluationUtils::update_confusion_matrix(confusion_matrix, true_label, predicted_statistics);
	}

	template <typename TTMatrix, typename TSampleIterator>
	TTMatrix& update_confusion_matrix(TTMatrix& confusion_matrix, const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
    {
        TStatistics true_statistics = EvaluationUtils::compute_true_statistics<TStatistics>(num_of_classes_, samples_start, samples_end);
        const TStatistics& predicted_statistics = compute_summed_statistics(samples_start, samples_end);
        return EvaluationUtils::update_confusion_matrix(confusion_matrix, true_statistics, predicted_statistics);
    }
};

template <typename TSplitPoint, typename TStatistics, typename TMatrix = Eigen::MatrixXd>
TreeUtilities<TSplitPoint, TStatistics, TMatrix> make_tree_utils(const Tree<TSplitPoint, TStatistics>& tree)
{
	return TreeUtilities<TSplitPoint, TStatistics, TMatrix>(tree);
}

template <typename TSplitPoint, typename TStatistics, typename TMatrix = Eigen::MatrixXd>
class ForestUtilities
{
	using ForestType = Forest<TSplitPoint, TStatistics>;

	const ForestType& forest_;
	std::vector<TreeUtilities<TSplitPoint, TStatistics, TMatrix>> tree_utils_vector_;
    size_type num_of_classes_;

public:
	using MatrixType = TMatrix;

	ForestUtilities(const ForestType& forest)
		: forest_(forest)
    {
        num_of_classes_ = forest_.cbegin()->get_root_iterator()->get_statistics().num_of_bins();
		for (auto tree_it = forest_.cbegin(); tree_it != forest_.cend(); ++tree_it) {
			TreeUtilities<TSplitPoint, TStatistics, TMatrix> tree_utils = TreeUtilities<TSplitPoint, TStatistics, TMatrix>(*tree_it);
			tree_utils_vector_.push_back(tree_utils);
		}
	}

	template <typename TSample>
	TStatistics compute_summed_statistics(const TSample& sample) const
	{
		TStatistics summed_statistics(num_of_classes_);
		for (auto tree_utils_it = tree_utils_vector_.cbegin(); tree_utils_it != tree_utils_vector_.cend(); ++tree_utils_it) {
			const TStatistics& tree_statistics = tree_utils_it->compute_statistics(sample);
            assert(num_of_classes_ == tree_statistics.num_of_bins());
//            summed_statistics.lazy_accumulate(tree_statistics.get_max_bin());
			summed_statistics.accumulate(tree_statistics);
        }
//        summed_statistics.finish_lazy_accumulation();
		return summed_statistics;
	}

	template <typename TSample>
	const TStatistics& compute_multiplied_statistics(const TSample& sample) const
	{
		TStatistics multiplied_statistics(num_of_classes_);
		for (size_type i = 0; i < num_of_classes_; ++i) {
			multiplied_statistics.get_histogram()[i] = 1;
		}
		for (auto tree_utils_it = tree_utils_vector_.cbegin(); tree_utils_it != tree_utils_vector_.cend(); ++tree_utils_it) {
			const TStatistics& tree_statistics = tree_utils_it->compute_statistics(sample);
			assert(num_of_classes_ == tree_statistics.num_of_bins());
			for (size_type i = 0; i < num_of_classes_; ++i) {
				multiplied_statistics.get_histogram()[i] *= tree_statistics.get_histogram()[i];
			}
		}
		return multiplied_statistics;
	}

	template <typename TSampleIterator>
	TStatistics compute_summed_statistics(const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
	{
		TStatistics summed_statistics(num_of_classes_);
		for (auto tree_utils_it = tree_utils_vector_.cbegin(); tree_utils_it != tree_utils_vector_.cend(); ++tree_utils_it) {
			const TStatistics& tree_statistics = tree_utils_it->compute_summed_statistics(samples_start, samples_end);
            assert(num_of_classes_ == tree_statistics.num_of_bins());
//            summed_statistics.lazy_accumulate(tree_statistics.get_max_bin());
			summed_statistics.accumulate(tree_statistics);
        }
//        summed_statistics.finish_lazy_accumulation();
		return summed_statistics;
	}

	template <typename TSampleIterator>
	TStatistics compute_multiplied_statistics(const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
	{
		TStatistics multiplied_statistics(num_of_classes_);
		for (size_type i = 0; i < num_of_classes_; ++i) {
			multiplied_statistics.get_histogram()[i] = 1;
		}
		for (auto tree_utils_it = tree_utils_vector_.cbegin(); tree_utils_it != tree_utils_vector_.cend(); ++tree_utils_it) {
			const TStatistics& tree_statistics = tree_utils_it->compute_multiplied_statistics(samples_start, samples_end);
			assert(num_of_classes_ == tree_statistics.num_of_bins());
			for (size_type i = 0; i < num_of_classes_; ++i) {
				multiplied_statistics.get_histogram()[i] *= tree_statistics.get_histogram()[i];
			}
		}
		return multiplied_statistics;
	}

	template <typename TSampleIterator>
	TMatrix compute_confusion_matrix(const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
	{
		TMatrix confusion_matrix(num_of_classes_, num_of_classes_);
        confusion_matrix.setZero();
        for (TSampleIterator sample_it = samples_start; sample_it != samples_end; ++sample_it) {
            update_confusion_matrix(confusion_matrix, *sample_it);
        }
		return confusion_matrix;
	}

	template <typename TTMatrix, typename TSample>
	TTMatrix& update_confusion_matrix(TTMatrix& confusion_matrix, const TSample& sample) const
    {
        size_type true_label = sample.get_label();
		const TStatistics& predicted_statistics = compute_summed_statistics(sample);
		return EvaluationUtils::update_confusion_matrix(confusion_matrix, true_label, predicted_statistics);
	}

	template <typename TTMatrix, typename TSampleIterator>
    TTMatrix& update_confusion_matrix(TTMatrix& confusion_matrix, const TSampleIterator& samples_start, const TSampleIterator& samples_end) const
    {
        TStatistics true_statistics = EvaluationUtils::compute_true_statistics<TStatistics>(num_of_classes_, samples_start, samples_end);
		const TStatistics& predicted_statistics = compute_summed_statistics(samples_start, samples_end);
		return EvaluationUtils::update_confusion_matrix(confusion_matrix, true_statistics, predicted_statistics);
	}
};

template <typename TSplitPoint, typename TStatistics, typename TMatrix = Eigen::MatrixXd>
ForestUtilities<TSplitPoint, TStatistics, TMatrix> make_forest_utils(const Forest<TSplitPoint, TStatistics>& forest)
{
	return ForestUtilities<TSplitPoint, TStatistics, TMatrix>(forest);
}

}
