//
//  ait.h
//  DistRandomForest
//
//  Created by Benjamin Hepp on 29/09/15.
//
//

#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <Eigen/Dense>

#include "logger.h"

namespace ait
{

//typedef std::int64_t size_type;
//using size_type = EIGEN_DEFAULT_DENSE_INDEX_TYPE;
using size_type = std::ptrdiff_t;
using scalar_type = double;

}

namespace std
{
    template <typename T>
    std::ostream& operator<<(std::ostream& sout, const std::vector<T>& container)
    {
        sout << "{";
        for (typename std::vector<T>::const_iterator it = container.begin(); it != container.end(); ++it)
        {
            sout << *it;
            if (it != container.cend())
            {
                sout << ", ";
            }
        }
        sout << "}";
    }
}
