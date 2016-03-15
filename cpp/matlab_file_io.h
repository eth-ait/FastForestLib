//
//  matlab_file_io.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

/*
* This code is adapted from MATLAB by MathWorks.
* Copyright 1984-2003 The MathWorks, Inc.
*/

#pragma once

#include <vector>
#include <map>
#include <cstdio>

#include <mat.h>

#include "image_weak_learner.h"


namespace ait
{

template <typename data_type = double, typename label_type = std::size_t>
std::vector<ait::Image<>> load_images_from_matlab_file(const std::string& filename, const std::string& data_array_name = "data", const std::string& label_array_name = "label")
{
    using ImageType = ait::Image<>;

    /*
    * Open file to get directory
    */
    MATFile* pmat = matOpen(filename.c_str(), "r");
    if (pmat == nullptr)
        throw std::runtime_error("Error opening file '" + filename + "'.");

    /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
    if (matClose(pmat) != 0)
        throw std::runtime_error("Error closing file '" + filename + "'.");
    pmat = matOpen(filename.c_str(), "r");
    if (pmat == nullptr)
        throw std::runtime_error("Error opening file '" + filename + "'.");

    mxArray* data_pa = matGetVariable(pmat, data_array_name.c_str());
    if (data_pa == nullptr)
        throw std::runtime_error("Error reading data array in file '" + filename + "'.");

    mxArray* label_pa = matGetVariable(pmat, label_array_name.c_str());
    if (label_pa == nullptr)
        throw std::runtime_error("Error reading label array in file '" + filename + "'.");

    /* Diagnose header pa */
    mwSize data_num_of_dimensions = mxGetNumberOfDimensions(data_pa);
    mwSize label_num_of_dimensions = mxGetNumberOfDimensions(label_pa);
    if (data_num_of_dimensions != 3 || label_num_of_dimensions != 3)
        throw std::runtime_error("Can only handle arrays with a dimension of 3.");

    const mwSize* data_dimensions = mxGetDimensions(data_pa);
    const mwSize* label_dimensions = mxGetDimensions(label_pa);

    int num_of_images = static_cast<int>(data_dimensions[2]);
    int width = static_cast<int>(data_dimensions[1]);
    int height = static_cast<int>(data_dimensions[0]);
    if (label_dimensions[2] != num_of_images || label_dimensions[1] != width || label_dimensions[0] != height)
        throw std::runtime_error("The label and data array must have the same dimensions");

    // Memory layout of MATLAB arrays: width x height x num_of_images.
    // The first dimension changes first, then second, then third.
    const double* data_ptr = mxGetPr(data_pa);
    const double* label_ptr = mxGetPr(label_pa);

    std::vector<ImageType> images;

    for (int i = 0; i < num_of_images; i++) {
        typename ImageType::DataMatrixType data_matrix(width, height);
        typename ImageType::LabelMatrixType label_matrix(width, height);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                data_matrix(x, y) = static_cast<data_type>(data_ptr[x + y * width + i * width * height]);
                label_matrix(x, y) = static_cast<label_type>(label_ptr[x + y * width + i * width * height]);
            }
        }
        ImageType image(data_matrix, label_matrix);
        images.push_back(std::move(image));
    }
    mxDestroyArray(data_pa);
    mxDestroyArray(label_pa);

    if (matClose(pmat) != 0)
        throw std::runtime_error("Error closing file '" + filename + "'.");

    return images;
}
    
template <typename TMatrix>
void write_arrays_to_matlab_file(const std::string& filename, const std::string& name, const std::vector<TMatrix>& arrays)
{
    /*
     * Open file to get directory
     */
    MATFile* pmat = matOpen(filename.c_str(), "w");
    if (pmat == nullptr) {
        throw std::runtime_error("Error opening file '" + filename + "'.");
    }
    
    /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
    if (matClose(pmat) != 0) {
        throw std::runtime_error("Error closing file '" + filename + "'.");
    }
    
    pmat = matOpen(filename.c_str(), "w");
    if (pmat == nullptr) {
        throw std::runtime_error("Error opening file '" + filename + "'.");
    }

    mxArray* cell_pa = mxCreateCellMatrix(1, arrays.size());
    if (cell_pa == nullptr) {
        throw std::runtime_error("Unable to create MATLAB cell array.");
    }

    for (auto it = arrays.cbegin(); it != arrays.cend(); ++it) {
        const TMatrix& array = *it;
        mxArray* array_pa = mxCreateDoubleMatrix(array.rows(), array.cols(), mxREAL);
        if (array_pa == nullptr) {
            throw std::runtime_error("Unable to create MATLAB matrix.");
        }
        double* array_ptr = mxGetPr(array_pa);
        for (int i = 0; i < array.rows(); ++i) {
            for (int j = 0; j < array.cols(); ++j) {
                array_ptr[i + j * array.rows()] = array(i, j);
            }
        }
        mxSetCell(cell_pa, it - arrays.cbegin(), array_pa);
    }

    if (matPutVariable(pmat, name.c_str(), cell_pa) != 0) {
        throw std::runtime_error("Unable to add cell array to .MAT file '" + filename + "'.'");
    }

    if (matClose(pmat) != 0) {
        throw std::runtime_error("Error closing file '" + filename + "'.");
    }
    
    // Also deallocates all the matrices
    mxDestroyArray(cell_pa);
}

template <typename TMatrix>
void write_arrays_to_matlab_file(const std::string& filename, const std::map<std::string, TMatrix>& array_map)
{
    /*
     * Open file to get directory
     */
    MATFile* pmat = matOpen(filename.c_str(), "w");
    if (pmat == nullptr) {
        throw std::runtime_error("Error opening file '" + filename + "'.");
    }
    
    /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
    if (matClose(pmat) != 0) {
        throw std::runtime_error("Error closing file '" + filename + "'.");
    }

    pmat = matOpen(filename.c_str(), "r");
    if (pmat == nullptr) {
        throw std::runtime_error("Error opening file '" + filename + "'.");
    }

    for (auto it = array_map.cbegin(); it != array_map.cend(); ++it) {
        const std::string& name = it->first;
        const TMatrix& array = it->second;
        mxArray* array_pa = mxCreateDoubleMatrix(array.rows(), array.cols(), mxREAL);
        if (array_pa == nullptr) {
            throw std::runtime_error("Unable to create MATLAB matrix.");
        }
        double* array_ptr = mxGetPr(array_pa);
        for (int i = 0; i < array.rows(); ++i) {
            for (int j = 0; j < array.cols(); ++j) {
                array_ptr[i * array.cols() + j] = array(i, j);
            }
        }
        if (matPutVariable(pmat, name.c_str(), array_pa)) {
            throw std::runtime_error("Unable to add array to .MAT file '" + filename + "'.'");
        }
        mxDestroyArray(array_pa);
    }

    if (matClose(pmat) != 0) {
        throw std::runtime_error("Error closing file '" + filename + "'.");
    }
}
    
template <typename TMatrix>
void write_array_to_matlab_file(const std::string& filename, const std::string& name, const TMatrix& array)
{
    std::map<std::string, TMatrix> array_map;
    array_map[name] = array;
    write_arrays_to_matlab_file(filename, array_map);
}

}
