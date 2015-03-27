#include "image_weak_learner.h"
#include "forest.h"
#include "tree.h"
#include "node.h"
#include "histogram_statistics.h"

int main(int argc, char **argv) {
  typedef AIT::Forest<ImageSplitPoint, HistogramStatistics> forest_type;

  forest_type forest;

  return 0;
}
