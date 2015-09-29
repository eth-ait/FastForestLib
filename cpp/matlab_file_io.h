/*
* This code is adapted from MATLAB by MathWorks.
* Copyright 1984-2003 The MathWorks, Inc.
*/

#pragma once

#include <vector>
#include <map>

#include <mat.h>

#include "image_weak_learner.h"


namespace ait
{

template <typename data_type = double, typename label_type = std::size_t>
std::vector<ait::Image> load_images_from_matlab_file(const std::string &filename, const std::string &data_array_name = "data", const std::string &label_array_name = "label")
{
    typedef ait::Image ImageType;

    /*
    * Open file to get directory
    */
    MATFile *pmat = matOpen(filename.c_str(), "r");
    if (pmat == nullptr)
        throw std::runtime_error("Error opening file '" + filename + "'.");

    /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
    if (matClose(pmat) != 0)
        throw std::runtime_error("Error closing file '" + filename + "'.");
    pmat = matOpen(filename.c_str(), "r");
    if (pmat == nullptr)
        throw std::runtime_error("Error opening file '" + filename + "'.");

    mxArray *data_pa = matGetVariable(pmat, data_array_name.c_str());
    if (data_pa == nullptr)
        throw std::runtime_error("Error reading data array in file '" + filename + "'.");

    mxArray *label_pa = matGetVariable(pmat, label_array_name.c_str());
    if (label_pa == nullptr)
        throw std::runtime_error("Error reading label array in file '" + filename + "'.");

    /* Diagnose header pa */
    mwSize data_num_of_dimensions = mxGetNumberOfDimensions(data_pa);
    mwSize label_num_of_dimensions = mxGetNumberOfDimensions(label_pa);
    if (data_num_of_dimensions != 3 || label_num_of_dimensions != 3)
        throw std::runtime_error("Can only handle arrays with a dimension of 3.");

    const mwSize *data_dimensions = mxGetDimensions(data_pa);
    const mwSize *label_dimensions = mxGetDimensions(label_pa);

    int num_of_images = static_cast<int>(data_dimensions[2]);
    int width = static_cast<int>(data_dimensions[1]);
    int height = static_cast<int>(data_dimensions[0]);
    if (label_dimensions[2] != num_of_images || label_dimensions[1] != width || label_dimensions[0] != height)
        throw std::runtime_error("The label and data array must have the same dimensions");

    // Memory layout of MATLAB arrays: width x height x num_of_images.
    // The first dimension changes first, then second, then third.
    const double *data_ptr = mxGetPr(data_pa);
    const double *label_ptr = mxGetPr(label_pa);

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

// TODO
//template <typename data_type=double>
//std::map<std::string, Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic>> load_matlab_file(const std::string &filename, const std::vector<std::string> &array_names) {
//    typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

//    std::map<std::string, MatrixType> array_map;

//    std::cout << "Reading file '" << filename << "'" << std::endl;

//    /*
//    * Open file to get directory
//    */
//    MATFile *pmat = matOpen(filename.c_str(), "r");
//    if (pmat == nullptr)
//        throw std::runtime_error("Error opening file '" + filename + "'.");

//    /*
//    * get directory of MAT-file
//    */
//    int	  ndir;
//    char **dir = matGetDir(pmat, &ndir);
//    if (dir == NULL)
//        throw std::runtime_error("Error reading directory of file '" + filename + "'.");
//    else {
//        std::cout << "Directory of '" << filename << "'" << std::endl;
//        for (int i = 0; i < ndir; i++)
//            std::cout << dir[i] << std::endl;
//    }
//    mxFree(dir);

//    /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
//    if (matClose(pmat) != 0)
//        throw std::runtime_error("Error closing file '" + filename + "'.");
//    pmat = matOpen(filename.c_str(), "r");
//    if (pmat == nullptr)
//        throw std::runtime_error("Error opening file '" + filename + "'.");

//    /* Get headers of all variables */
//    std::cout << std::endl << "Examining the header for each variable:" << std::endl;
//    for (const std::string &name : array_names) {
//        mxArray *pa = matGetVariable(pmat, name.c_str());
//        if (pa == nullptr)
//            throw std::runtime_error("Error reading in file '" + filename + "'.");
//        /* Diagnose header pa */
//        mwSize num_of_dimensions = mxGetNumberOfDimensions(pa);
//        std::cout << "According to its header, array '" << name << "' has " << num_of_dimensions << " dimensions" << std::endl;
//        if (num_of_dimensions != 2)
//            throw std::runtime_error("Can only handle arrays with a dimension of 2.");
//        const double *data_ptr = mxGetPr(pa);
//        MatrixType matrix(mxGetM(pa), mxGetN(pa));
//        for (int row = 0; row < matrix.rows(); row++)
//        for (int col = 0; col < matrix.cols(); col++)
//            matrix(row, col) = data_ptr[row + col * matrix.rows()];
//        mxDestroyArray(pa);
//        array_map[name] = std::move(matrix);
//    }

//    if (matClose(pmat) != 0)
//        throw std::runtime_error("Error closing file '" + filename + "'.");

//    return array_map;
//}

}
