#ifndef AITDistributedRandomForest_statistics_h
#define AITDistributedRandomForest_statistics_h

#include <vector>
#include <cmath>
#include <numeric>

namespace AIT {

	/// @brief A histogram over the classes of samples.
	template <typename TSample, typename t_entropy_type = double, typename count_type = int>
	class HistogramStatistics {
	public:
//        typedef int count_type;
		typedef t_entropy_type entropy_type;
		typedef typename TSample::label_type label_type;
		typedef typename std::vector<count_type>::size_type size_type;

	private:
		std::vector<count_type> histogram_;
		count_type num_of_samples_;

	public:
		// TODO: unused
		/// @brief Create an empty histogram.
		HistogramStatistics()
		: num_of_samples_(0) {}
      
		// TODO: unused
		/// @brief Create an empty histogram.
		/// @param num_of_classes The number of classes.
		HistogramStatistics(size_type num_of_classes)
		: histogram_(num_of_classes, 0),
			num_of_samples_(0) {}

		/// @brief Create a histogram from a vector of counts per class.
		HistogramStatistics(const std::vector<count_type> &histogram)
		: histogram_(histogram),
        num_of_samples_(std::accumulate(histogram.cbegin(), histogram.cend())) {}

		inline void Accumulate(const TSample &sample) {
			label_type label = sample.GetLabel();
			if (label >= histogram_.size())
				histogram_.resize(label + 1);
			histogram_[label]++;
			num_of_samples_++;
		}

		/// @brief Return the numbers of samples contributing to the histogram.
		count_type NumOfSamples() const {
			return num_of_samples_;
		}

		/// @brief Return the vector of counts per class.
		const std::vector<count_type> & GetHistogram() const {
			return histogram_;
		}

		/// @return: The Shannon entropy of the histogram.
		inline const entropy_type Entropy() const {
			entropy_type entropy = 0;
			for (auto it=histogram_.cbegin(); it != histogram_.cend(); it++) {
				const entropy_type count = static_cast<entropy_type>(*it);
				if (count > 0) {
					const entropy_type relative_count = count / num_of_samples_;
					entropy -= relative_count * std::log2(relative_count);
				}
			}
			return entropy;
		}

	};

}

#endif
