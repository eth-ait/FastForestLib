//
//  ait.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 29/09/15.
//
//

#pragma once

#include <cstdint>
#include <Eigen/Dense>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace ait
{

//typedef std::int64_t size_type;
using size_type = EIGEN_DEFAULT_DENSE_INDEX_TYPE;
using scalar_type = double;

}
