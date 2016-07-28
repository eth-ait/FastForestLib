//
//  hdf5_file_io.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

#include "H5Cpp.h"

namespace ait
{

H5::DataSet* create_hdf5_dataset(H5::H5File* file, const std::string& name,
                                 const std::vector<int64_t>& dimensions, const H5::DataType& data_type) {
    using namespace H5;

//    // Create property list for a dataset and set up fill values
//    Scalar fillvalue(0);
//    DSetCreatPropList plist;
//    plist.setFillValue(PredType::NATIVE_INT, &fillvalue);

    // Create dataspace for the dataset in the file
    hsize_t* fdim = new hsize_t[dimensions.size()];
    for (int i = 0; i < dimensions.size(); ++i) {
        fdim[i] = dimensions[i];
    }
    DataSpace fspace(dimensions.size(), fdim);

    // Create dataset and write it into the file
    DataSet* dataset = new DataSet(file->createDataSet(name, data_type, fspace));

    return dataset;
}

template <typename T>
H5::DataType get_hdf5_data_type();

template <>
H5::DataType get_hdf5_data_type<std::int8_t>() {
    return H5::PredType::NATIVE_INT8;
}

template <>
H5::DataType get_hdf5_data_type<std::int16_t>() {
    return H5::PredType::NATIVE_INT16;
}

template <>
H5::DataType get_hdf5_data_type<std::int32_t>() {
    return H5::PredType::NATIVE_INT32;
}

template <>
H5::DataType get_hdf5_data_type<std::int64_t>() {
    return H5::PredType::NATIVE_INT64;
}

template <>
H5::DataType get_hdf5_data_type<float>() {
    return H5::PredType::NATIVE_FLOAT;
}

template <>
H5::DataType get_hdf5_data_type<double>() {
    return H5::PredType::NATIVE_DOUBLE;
}

template <typename TMatrix>
void write_arrays_to_hdf5_file(const std::string& filename, const std::map<std::string, TMatrix>& array_map)
{
    using namespace H5;
    H5File* file = new H5File(filename, H5F_ACC_TRUNC);

    DataType data_type = get_hdf5_data_type<typename TMatrix::Scalar>();

    for (auto it = array_map.cbegin(); it != array_map.cend(); ++it) {
        const std::string& name = it->first;
        const TMatrix& array = it->second;

        std::vector<std::int64_t> dimensions = {array.rows(), array.cols()};
        DataSet* dataset = create_hdf5_dataset(file, name, dimensions, data_type);

        DataSpace array_space = dataset->getSpace();
        DataSpace dataset_space = dataset->getSpace();
        dataset->write(reinterpret_cast<const void*>(&array(0, 0)), data_type, array_space, dataset_space);
    }

    file->close();
}

template <typename TMatrix>
void write_array_to_hdf5_file(const std::string& filename, const std::string& name, const TMatrix& array)
{
    std::map<std::string, TMatrix> array_map;
    array_map[name] = array;
    write_arrays_to_hdf5_file(filename, array_map);
}

}
