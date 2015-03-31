#include <strstream>
#include <vector>

#include "forest.h"
#include "tree.h"
#include "node.h"
#include "histogram_statistics.h"
#include "image_weak_learner.h"
#include "eigen_matrix_io.h"
#include "matlab_file_io.cpp"


typedef AIT::ImageSample<> SampleType;
typedef AIT::Forest<AIT::ImageSplitPoint<>, AIT::HistogramStatistics<AIT::ImageSample<>>> forest_type;
typedef std::vector<AIT::ImageSample<>>::iterator iterator_type;

template <typename TSample, typename TWeakLearner, typename TTrainingParameters>
class TrainingOperation {
public:
	typedef AIT::Node<typename TWeakLearner::SplitPoint, typename TWeakLearner::Statistics> NodeType;
	typedef typename std::vector<TSample>::iterator SampleIterator;

private:
	std::vector<TSample> samples_;
	TWeakLearner weak_learner_;
	TTrainingParameters training_parameters_;

public:
	TrainingOperation(const std::vector<TSample> &samples, const TWeakLearner &weak_learner, const TTrainingParameters &training_parameters)
		: samples_(samples), weak_learner_(weak_learner), training_parameters_(training_parameters) {}

	void TrainRecursive(const NodeType &node, const SampleIterator i_start, const SampleIterator i_end, int current_depth = 1) const {
		std::ostringstream o_stream;
		for (int i = 0; i < current_depth; i++)
			o_stream << " ";
		const std::string prefix = o_stream.str();
		std::cout << prefix << "depth " << current_depth << std::endl;

		// assign statistics to node
		TWeakLearner::Statistics = weak_learner_.ComputeStatistics(samples_.cbegin(), samples_.cend());
		statistics = self._weak_learner_context.compute_statistics(sample_indices);
		node.SetStatistics(statistics);

		// stop splitting the node if the minimum number of samples has been reached
		if (i_end - i_start < training_parameters.MinimumNumOfSamples()) {
			//node.leaf_node = True
			std::cout << prefix << "Minimum number of samples. Stopping." << std::endl;
			return;
		}

		// stop splitting the node if it is a leaf node
		/*if (node.left_child is None) {
			node.leaf_node = True
			print("{}Reached leaf node. Stopping.".format(prefix))
			return
		}*/

		std::vector<TSplitNode> split_points = weak_learner.SampleSplitPoints(i_start, i_end, training_parameters_.NumOfFeatures(), training_parameters_.NumOfThresholds());

		// TODO: distribute features and thresholds to ranks > 0

		// compute the statistics for all feature and threshold combinations
		SplitStatistics<TWeakLearner::Statistics> split_statistics = weak_learner.ComputeSplitStatistics(i_start, i_end, split_points);

		// TODO: send statistics to rank 0
		// send split_statistics.get_buffer()

		// TODO: receive statistics from rank > 0
		// for received statistics
			// split_statistics.accumulate(statistics)

		// find the best feature(only on rank 0)
		std::tuple<TWeakLearner::size_type, TWeakLearner::entropy_type> best_split_point_tuple = weak_learner.SelectBestSplitPoint(statistics, split_statistics);

		// TODO: send best feature, threshold and information gain to ranks > 0

		TWeakLearner::entropy_type best_information_gain = std::get<1>(best_split_point_tuple);
		// TODO: move criterion into trainingContext ?
		// stop splitting the node if the best information gain is below the minimum information gain
		if (best_information_gain < training_parameters.MinimumInformationGain()) {
			//node.leaf_node = True
			std::cout << prefix << "Too little information gain. Stopping." << std::endl;
			return;
		}

		// partition sample_indices according to the selected feature and threshold.
		// i.e.sample_indices[:i_split] will contain the left child indices
		// and sample_indices[i_split:] will contain the right child indices
		TWeakLearner::size_type best_split_point_index = std::get<0>(best_split_point_tuple);
		TWeakLearner::SplitPoint best_split_point = split_points.GetSplitPoint(best_split_point_index);
		TWeakLearner::Iterator i_split = i_start + weak_learner_context.Partition(i_start, i_end, best_split_point);

		node.SetSplitPoint(best_split_point);

		// TODO: can we reuse computed statistics from split_point_context ? ? ?
		//left_child_statistics = None
		//right_child_statistics = None

		// train left and right child
		//print("{}Going left".format(prefix))
		train_recursive(node.LeftChild(), i_start, i_split, current_depth + 1);
		//print("{}Going right".format(prefix))
		self.train_recursive(node.RightChild(), i_split, i_end, current_depth + 1);
	}

	/*def train_forest(self, sample_indices, training_context, training_parameters) :
	forest = Forest()
for i in xrange(training_parameters.numOfTrees) :
# TODO : perform bagging on the samples
	tree = ArrayTree(training_parameters.maximumDepth)
	self.train_tree(tree, sample_indices, training_context, training_parameters)
	forest.append(tree)
	return forest

	def train_tree(self, tree, sample_indices, training_context, training_parameters) :
	rf_operation = self._TrainingOperation(sample_indices, training_context, training_parameters)
	i_start = 0
	i_end = len(sample_indices)
	print("Training tree")
	rf_operation.train_recursive(tree.root, i_start, i_end)
	return tree*/
};

int main(int argc, const char *argv[]) {
	try {
		std::vector<std::string> array_names;
		array_names.push_back("data");
		array_names.push_back("labels");
		std::map<std::string, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> array_map = LoadMatlabFile("trainingData.mat", array_names);

		forest_type forest;
		AIT::ImageWeakLearner<AIT::HistogramStatistics<AIT::ImageSample<>, int, double, std::size_t>, iterator_type> iwl;
	}
	catch (std::runtime_error error) {
		std::cerr << "Runtime exception occured" << std::endl;
		std::cerr << error.what() << std::endl;
	}

	return 0;
}
