//
//  training.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 06/10/15.
//
//

#pragma once

namespace ait
{

struct TrainingParameters
{
#if AIT_TESTING
    int num_of_trees = 3;
    int tree_depth = 12;
#else
    int num_of_trees = 3;
    int tree_depth = 12;
#endif
    int minimum_num_of_samples = 100;
    double minimum_information_gain = 0.0;
#if AIT_MULTI_THREADING
    int num_of_threads = -1;
#endif
};

}
