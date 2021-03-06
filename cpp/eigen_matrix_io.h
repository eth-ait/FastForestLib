//
//  eigen_matrix_io.h
//  DistRandomForest
//
//  Created by Benjamin Hepp.
//
//

#pragma once

#include <iostream>
#include <fstream>
#include <memory>

#include <Eigen/Dense>

namespace ait
{

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
std::unique_ptr<Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>> load_matrix(const std::string& filename)
{
	using MatrixType = Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;

	std::ifstream input(filename.c_str(), std::ios::binary);
	if (input.fail()) {
		throw std::runtime_error("Cannot open matrix file '" + filename + "'.");
	}

	int rows;
	input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
	if (!input)
		throw std::runtime_error("Invalid matrix input file. Number of rows is not specified.");

	int cols;
	input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
	if (!input)
		throw std::runtime_error("Invalid matrix input file. Number of columns is not specified.");

	std::unique_ptr<MatrixType> m_ptr = std::unique_ptr<MatrixType>(new MatrixType(rows, cols));

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			Scalar value;
			input.read(reinterpret_cast<char*>(&value), sizeof(Scalar));
			/*if (!input)
				throw std::runtime_error("Could not read data.");*/
			(*m_ptr)(row, col) = value;
		}
	}

	input.close();

	return std::move(m_ptr);
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
void save_matrix(const std::string& filename, const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>& matrix)
{
	using MatrixType = Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;

	std::ofstream output(filename);
	if (output.fail()) {
		throw std::runtime_error("Cannot open matrix file '" + filename + "' for writing.");
	}

	int rows = matrix.rows();
	output.write(reinterpret_cast<char*>(&rows), sizeof(rows));
	if (!output)
		throw std::runtime_error("Cannot write to matrix file.");

	int cols = matrix.cols();
	output.write(reinterpret_cast<char*>(&cols), sizeof(cols));
	if (!output)
		throw std::runtime_error("Cannot write to matrix file.");

	std::unique_ptr<MatrixType> m_ptr = std::unique_ptr<MatrixType>(new MatrixType(rows, cols));

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			Scalar value = (*m_ptr)(row, col);
			output.write(reinterpret_cast<char*>(&value), sizeof(Scalar));
			/*if (!output)
				throw std::runtime_error("Could not write data.");*/
		}
	}

	output.close();
}

}
