#include <iostream>
#include <memory>

#include <Eigen/Dense>

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
std::unique_ptr<Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>> LoadMatrix(const std::string &filename) {
	typedef Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> MatrixType;

	std::ifstream input(filename.c_str(), std::ios::binary);
	if (input.fail()) {
		throw std::runtime_error("Cannot open matrix file '" + filename + "'.");
	}

	int rows;
	input.read(&rows, sizeof(rows));
	if (!input)
		throw std::runtime_error("Invalid matrix input file. Number of rows is not specified.");

	int cols;
	input.read(&cols, sizeof(cols));
	if (!input)
		throw std::runtime_error("Invalid matrix input file. Number of columns is not specified.");

	std::unique_ptr<MatrixType> m_ptr = std::unique_ptr<MatrixType>(new MatrixType(rows, cols));

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			Scalar value;
			input.read(&value, sizeof(Scalar));
			/*if (!input)
				throw std::runtime_error("Could not read data.");*/
			(*m_ptr)(row, col) = value;
		}
	}

	input.close();

	return std::move(m_ptr);
}

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
void SaveMatrix(const std::string &filename, const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> &matrix) {
	typedef Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> MatrixType;

	std::ofstream output(filename);
	if (output.fail) {
		throw std::runtime_error("Cannot open matrix file '" + filename + "' for writing.");
	}

	int rows = matrix.rows();
	output.write(&rows, sizeof(rows));
	if (!output)
		throw std::runtime_error("Cannot write to matrix file.");

	int cols = matrix.cols();
	output.write(&cols, sizeof(cols));
	if (!output)
		throw std::runtime_error("Cannot write to matrix file.");

	std::unique_ptr<MatrixType> m_ptr = std::unique_ptr<MatrixType>(new MatrixType(rows, cols));

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			Scalar value = (*m_ptr)(row, col);
			output.write(&value, sizeof(Scalar));
			/*if (!output)
				throw std::runtime_error("Could not write data.");*/
		}
	}

	output.close();
}
