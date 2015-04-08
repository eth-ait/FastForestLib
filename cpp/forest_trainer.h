#ifndef AITDistributedRandomForest_forest_trainer_h
#define AITDistributedRandomForest_forest_trainer_h

#include <iostream>
#include <sstream>

#include "forest.h"
#include "weak_learner.h"

namespace AIT {

    class TrainingParameters {
    public:
        int NumOfTrees() const {
            return 1;
        }
        int TreeDepth() const {
            return 10;
        }
        int MinimumNumOfSamples() const {
            return 100;
        }
        int NumOfThresholds() const {
            return 10;
        }
        int NumOfFeatures() const {
            return 10;
        }
        double MinimumInformationGain() const {
            return 0.0;
        }
    };

	template <typename TSample, typename TWeakLearner, typename TTrainingParameters, typename TRandomEngine>
	class ForestTrainer {
	public:
        typedef typename TWeakLearner::SplitPoint SplitPoint;
        typedef typename TWeakLearner::Statistics Statistics;
        typedef typename TWeakLearner::Iterator SampleIterator;
        typedef typename AIT::Tree<SplitPoint, Statistics>::NodeIterator NodeIterator;

	private:
		const TWeakLearner weak_learner_;
		const TTrainingParameters training_parameters_;

	public:
		ForestTrainer(const TWeakLearner &weak_learner, const TTrainingParameters &training_parameters)
			: weak_learner_(weak_learner), training_parameters_(training_parameters)
        {}

        void TrainTreeRecursive(NodeIterator node_iter, SampleIterator i_start, SampleIterator i_end, TRandomEngine &rnd_engine, int current_depth = 0) const
        {
            // TODO: Remove io operations
			std::ostringstream o_stream;
			for (int i = 0; i < current_depth; i++)
				o_stream << " ";
			const std::string prefix = o_stream.str();
			std::cout << prefix << "depth: " << current_depth << ", samples: " << (i_end - i_start) << std::endl;

			// assign statistics to node
			typename TWeakLearner::Statistics statistics = weak_learner_.ComputeStatistics(i_start, i_end);
			node_iter->SetStatistics(statistics);

			// stop splitting the node if the minimum number of samples has been reached
			if (i_end - i_start < training_parameters_.MinimumNumOfSamples()) {
				//node.leaf_node = True
				std::cout << prefix << "Minimum number of samples. Stopping." << std::endl;
				return;
			}

			// stop splitting the node if it is a leaf node
            if (node_iter.IsLeafNode()) {
                std::cout << prefix << "Reached leaf node. Stopping." << std::endl;
                return;
            }

			std::vector<typename TWeakLearner::SplitPoint> split_points = weak_learner_.SampleSplitPoints(i_start, i_end, training_parameters_.NumOfFeatures(), training_parameters_.NumOfThresholds(), rnd_engine);

			// TODO: distribute features and thresholds to ranks > 0

			// compute the statistics for all feature and threshold combinations
			SplitStatistics<typename TWeakLearner::Statistics> split_statistics = weak_learner_.ComputeSplitStatistics(i_start, i_end, split_points);

			// TODO: send statistics to rank 0
			// send split_statistics.get_buffer()

			// TODO: receive statistics from rank > 0
			// for received statistics
				// split_statistics.accumulate(statistics)

			// find the best feature(only on rank 0)
			std::tuple<typename TWeakLearner::size_type, typename TWeakLearner::entropy_type> best_split_point_tuple = weak_learner_.FindBestSplitPoint(statistics, split_statistics);

			// TODO: send best feature, threshold and information gain to ranks > 0

			typename TWeakLearner::entropy_type best_information_gain = std::get<1>(best_split_point_tuple);
			// TODO: move criterion into trainingContext ?
			// stop splitting the node if the best information gain is below the minimum information gain
			if (best_information_gain < training_parameters_.MinimumInformationGain()) {
				//node.leaf_node = True
				std::cout << prefix << "Too little information gain. Stopping." << std::endl;
				return;
			}

			// partition sample_indices according to the selected feature and threshold.
			// i.e.sample_indices[:i_split] will contain the left child indices
			// and sample_indices[i_split:] will contain the right child indices
			typename TWeakLearner::size_type best_split_point_index = std::get<0>(best_split_point_tuple);
			typename TWeakLearner::SplitPoint best_split_point = split_points[best_split_point_index];
			typename TWeakLearner::Iterator i_split = weak_learner_.Partition(i_start, i_end, best_split_point);

			node_iter->SetSplitPoint(best_split_point);

			// TODO: can we reuse computed statistics from split_point_context ? ? ?
			//left_child_statistics = None
			//right_child_statistics = None

			// train left and right child
			//print("{}Going left".format(prefix))
			TrainTreeRecursive(node_iter.LeftChild(), i_start, i_split, rnd_engine, current_depth + 1);
			//print("{}Going right".format(prefix))
			TrainTreeRecursive(node_iter.RightChild(), i_split, i_end, rnd_engine, current_depth + 1);
		}
        
        Tree<SplitPoint, Statistics> TrainTree(std::vector<TSample> &samples) const
        {
            TRandomEngine rnd_engine;
            return TrainTree(samples, rnd_engine);
        }
        

        Tree<SplitPoint, Statistics> TrainTree(std::vector<TSample> &samples, TRandomEngine &rnd_engine) const
        {
            Tree<SplitPoint, Statistics> tree(training_parameters_.TreeDepth());
            TrainTreeRecursive(tree.GetRoot(), samples.begin(), samples.end(), rnd_engine);
            return tree;
        }
        
        Forest<SplitPoint, Statistics> TrainForest(std::vector<TSample> &samples) const
        {
            TRandomEngine rnd_engine;
            return TrainForest(samples, rnd_engine);
        }

        Forest<SplitPoint, Statistics> TrainForest(std::vector<TSample> &samples, TRandomEngine &rnd_engine) const
        {
            Forest<SplitPoint, Statistics> forest;
            for (int i=0; i < training_parameters_.NumOfTrees(); i++) {
                Tree<SplitPoint, Statistics> tree = TrainTree(samples, rnd_engine);
                forest.AddTree(std::move(tree));
            }
            return forest;
        }

	};

}

#endif
