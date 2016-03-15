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
#include <chrono>
#include <Eigen/Dense>

#include "logger.h"

namespace ait
{

//typedef std::int64_t size_type;
//using size_type = EIGEN_DEFAULT_DENSE_INDEX_TYPE;
using size_type = std::ptrdiff_t;
using int_type = std::int64_t;
using scalar_type = double;

double compute_elapsed_seconds(const std::chrono::high_resolution_clock::time_point& start_time) {
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = stop_time - start_time;
    auto period = std::chrono::high_resolution_clock::period();
    double elapsed_milliseconds = duration.count() * period.num / static_cast<double>(period.den);
    return elapsed_milliseconds;
}

double compute_elapsed_milliseconds(const std::chrono::high_resolution_clock::time_point& start_time) {
    return 1000.0 * compute_elapsed_seconds(start_time);
}

}
