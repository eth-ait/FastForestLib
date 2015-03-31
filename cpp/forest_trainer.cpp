#include "forest.h"
#include "tree.h"
#include "node.h"
#include "histogram_statistics.h"
#include "image_weak_learner.h"


int main(int argc, const char *argv[]) {
  typedef AIT::Forest<AIT::ImageSplitPoint, AIT::HistogramStatistics<>> forest_type;
  typedef std::vector<double>::iterator iterator_type;

  forest_type forest;
  AIT::ImageWeakLearner<iterator_type, AIT::HistogramStatistics<int, double, int>> iwl;

  return 0;
}
