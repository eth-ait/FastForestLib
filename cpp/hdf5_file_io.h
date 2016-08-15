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

template <typename T>
class HDF5DataSpace;

template <typename T>
class HDF5Dataset {
	H5::DataType data_type_;
	std::shared_ptr<H5::DataSet> dataset_;
	std::vector<int64_t> dimensions_;

	HDF5Dataset(H5::H5File *file, const std::string& name, const std::vector<int64_t>& dimensions)
	: data_type_(get_hdf5_data_type<T>()), dimensions_(dimensions) {
		// Create dataspace for the dataset in the file
		hsize_t* fdim = new hsize_t[dimensions.size()];
		for (int i = 0; i < dimensions.size(); ++i) {
			fdim[i] = dimensions[i];
		}
		H5::DataSpace fspace(dimensions.size(), fdim);

		// Create dataset and write it into the file
		dataset_ = std::make_shared<H5::DataSet>(file->createDataSet(name, data_type_, fspace));
	}

public:
	friend class HDF5File;
	friend class HDF5DataSpace<T>;

	HDF5Dataset()
	: data_type_(get_hdf5_data_type<T>()) {
	}

	HDF5DataSpace<T> getDataSpace() {
		return HDF5DataSpace<T>(this, dataset_->getSpace());
	}

	const std::vector<int64_t>& getDimensions() const {
		return dimensions_;
	}
};

template <typename T>
class HDF5DataSpace {
	H5::DataType data_type_;
	HDF5Dataset<T> *dataset_;
	H5::DataSpace space_;

public:
	HDF5DataSpace()
	: data_type_(get_hdf5_data_type<T>()), dataset_(nullptr) {
	}

	HDF5DataSpace(HDF5Dataset<T> *dataset, H5::DataSpace space)
	: data_type_(get_hdf5_data_type<T>()),
	  dataset_(dataset), space_(space) {
	}

	void selectAll() {
		space_.selectAll();
	}

	void selectFirstDimensionSlice(int l) {
		selectDimensionSlice(0, l);
	}

	void selectLastDimensionSlice(int n) {
		selectDimensionSlice(dataset_->getDimensions().size() - 1, n);
	}

	void selectDimensionSlice(int dim, int index) {
		space_.selectNone();
		int num_dimensions = dataset_->getDimensions().size();
		std::vector<hsize_t> count(num_dimensions);
		std::vector<hsize_t> start(num_dimensions);
		std::vector<hsize_t> stride(num_dimensions);
		std::vector<hsize_t> block(num_dimensions);
		for (int i = 0; i < num_dimensions; ++i) {
			if (i == dim) {
				count[i] = 1;
				start[i] = index;
			} else {
				count[i] = dataset_->getDimensions()[i];
				start[i] = 0;
			}
			stride[i] = 1;
			block[i] = 1;
		}
		space_.selectHyperslab(H5S_SELECT_SET, &count[0], &start[0], &stride[0], &block[0]);
	}

	int64_t getSelectionSize() const {
		return space_.getSelectNpoints();
	}

	H5::DataSpace makeFlatMemorySpace(int64_t memory_size) {
		const hsize_t dims[1] = {static_cast<hsize_t>(memory_size)};
		return H5::DataSpace(1, dims);
	}

	void writeData(const std::vector<T> &data) {
		writeData(reinterpret_cast<const void*>(&data[0]));
	}

	void writeData(const T *data) {
		writeData(reinterpret_cast<const void*>(data));
	}

	void writeData(const void *data) {
		H5::DataSpace mem_space = makeFlatMemorySpace(getSelectionSize());
		dataset_->dataset_->write(data, data_type_, mem_space, space_);
	}

};

class HDF5File {
	H5::H5File *file_;
	bool open_;

public:
	HDF5File() {
		file_ = nullptr;
		open_ = false;
	}

	HDF5File(const std::string& filename, unsigned int mode=H5F_ACC_TRUNC) {
		open(filename, mode);
	}

	bool isOpen() const {
		return open_;
	}

	void open(const std::string& filename, unsigned int mode=H5F_ACC_TRUNC) {
		if (open_) {
			close();
		}
		file_ = new H5::H5File(filename, mode);
		open_ = true;
	}

	~HDF5File() {
		if (open_) {
			close();
		}
	}

	void close() {
		file_->close();
		open_ = false;
		delete file_;
	}

	template <typename T>
	HDF5Dataset<T> createDataset(const std::string& name, const std::vector<int64_t>& dimensions) {
		return HDF5Dataset<T>(file_, name, dimensions);
	}

	template <typename T>
	HDF5Dataset<T> createDataset(const std::string& name, int rows, int cols) {
		const std::vector<int64_t> dimensions = {rows, cols};
		return createDataset<T>(name, dimensions);
	}

	template <typename T>
	HDF5Dataset<T> createDataset(const std::string& name, int L, int M, int N) {
		const std::vector<int64_t> dimensions = {L, M, N};
		return createDataset<T>(name, dimensions);
	}
};

template <typename TMatrix>
void write_arrays_to_hdf5_file(const std::string& filename, const std::map<std::string, TMatrix>& array_map)
{
    HDF5File file(filename);
    for (auto it = array_map.cbegin(); it != array_map.cend(); ++it) {
        const std::string& name = it->first;
        const TMatrix& array = it->second;
        using data_type = typename TMatrix::Scalar;
        HDF5Dataset<data_type> dataset = file.createDataset<data_type>(name, array.rows(), array.cols());
        HDF5DataSpace<data_type> space = dataset.getDataSpace();
        space.selectAll();
        space.writeData(reinterpret_cast<const void*>(&array(0, 0)));
    }
}

template <typename TMatrix>
void write_array_to_hdf5_file(const std::string& filename, const std::string& name, const TMatrix& array)
{
    std::map<std::string, TMatrix> array_map;
    array_map[name] = array;
    write_arrays_to_hdf5_file(filename, array_map);
}

}
