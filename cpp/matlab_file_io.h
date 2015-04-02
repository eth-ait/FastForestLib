/*
* This code is adapted from MATLAB by MathWorks.
* Copyright 1984-2003 The MathWorks, Inc.
*/

#ifndef AIT_matlab_file_io_h
#define AIT_matlab_file_io_h

#include <vector>
#include <map>

#include "mat.h"

#include "image_weak_learner.h"


namespace AIT {

    std::vector<AIT::Image<> > LoadImagesFromMatlabFile(const std::string &filename, const std::string &data_array_name = "data", const std::string &label_array_name = "label") {
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

        int num_of_images = data_dimensions[2];
        int width = data_dimensions[1];
        int height = data_dimensions[0];
        if (label_dimensions[2] != num_of_images || label_dimensions[1] != width || label_dimensions[0] != height)
            throw std::runtime_error("The label and data array must have the same dimensions");

        // Memory layout of MATLAB arrays: width x height x num_of_images.
        // The first dimension changes first, then second, then third.
        const double *data_ptr = mxGetPr(data_pa);
        const double *label_ptr = mxGetPr(label_pa);

        std::vector<AIT::Image<> > images;

        for (int i = 0; i < num_of_images; i++) {
            typename AIT::Image<>::DataMatrixType data_matrix(width, height);
            typename AIT::Image<>::LabelMatrixType label_matrix(width, height);
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    data_matrix(x, y) = data_ptr[x + y * width + i * width * height];
                    label_matrix(x, y) = label_ptr[x + y * width + i * width * height];
                }
            }
            AIT::Image<> image(data_matrix, label_matrix);
            images.push_back(std::move(image));
        }
        mxDestroyArray(data_pa);
        mxDestroyArray(label_pa);

        if (matClose(pmat) != 0)
            throw std::runtime_error("Error closing file '" + filename + "'.");

        return images;
    }

    int diagnose(const std::string &filename) {
        std::cout << "Reading file '" << filename << "'" << std::endl;

        /*
        * Open file to get directory
        */
        MATFile *pmat = matOpen(filename.c_str(), "r");
        if (pmat == nullptr)
            throw std::runtime_error("Error opening filename '" + filename + "'.");

        /*
        * get directory of MAT-file
        */
        int	  ndir;
        char **dir = matGetDir(pmat, &ndir);
        if (dir == NULL)
            throw std::runtime_error("Error reading directory of filename '" + filename + "'.");
        else {
            std::cout << "Directory of '" << filename << "'" << std::endl;
            for (int i = 0; i < ndir; i++)
                std::cout << dir[i] << std::endl;
        }
        mxFree(dir);

        /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
        if (matClose(pmat) != 0)
            throw std::runtime_error("Error closing file '" + filename + "'.");
        pmat = matOpen(filename.c_str(), "r");
        if (pmat == nullptr)
            throw std::runtime_error("Error opening file '" + filename + "'.");

        /* Get headers of all variables */
        std::cout << std::endl << "Examining the header for each variable:" << std::endl;
        for (int i = 0; i < ndir; i++) {
            const char *name;
            mxArray *pa = matGetNextVariableInfo(pmat, &name);
            if (pa == nullptr)
                throw std::runtime_error("Error reading in file '" + filename + "'.");
            /* Diagnose header pa */
            std::cout << "According to its header, array '" << name << "' has " << mxGetNumberOfDimensions(pa) << " dimensions" << std::endl;
            if (mxIsFromGlobalWS(pa))
                std::cout << "  and was a global variable when saved" << std::endl;
            else
                std::cout << "  and was a local variable when saved" << std::endl;
            mxDestroyArray(pa);
        }

        /* Reopen file to read in actual arrays. */
        if (matClose(pmat) != 0)
            throw std::runtime_error("Error closing file '" + filename + "'.");
        pmat = matOpen(filename.c_str(), "r");
        if (pmat == nullptr)
            throw std::runtime_error("Error opening file '" + filename + "'.");

        /* Read in each array. */
        std::cout << std::endl << "Reading in the actual array contents:" << std::endl;
        for (int i = 0; i < ndir; i++) {
            const char *name;
            mxArray *pa = matGetNextVariable(pmat, &name);
            if (pa == nullptr)
                throw std::runtime_error("Error reading in file '" + filename + "'.");
            mxDestroyArray(pa);
        }

        if (matClose(pmat) != 0)
            throw std::runtime_error("Error closing file '" + filename + "'.");

        std::cout << "Done" << std::endl;

        return 0;
    }

    template <typename data_type=double>
    std::map<std::string, Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic>> LoadMatlabFile(const std::string &filename, const std::vector<std::string> &array_names) {
        typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

        std::map<std::string, MatrixType> array_map;

        std::cout << "Reading file '" << filename << "'" << std::endl;

        /*
        * Open file to get directory
        */
        MATFile *pmat = matOpen(filename.c_str(), "r");
        if (pmat == nullptr)
            throw std::runtime_error("Error opening file '" + filename + "'.");

        /*
        * get directory of MAT-file
        */
        int	  ndir;
        char **dir = matGetDir(pmat, &ndir);
        if (dir == NULL)
            throw std::runtime_error("Error reading directory of file '" + filename + "'.");
        else {
            std::cout << "Directory of '" << filename << "'" << std::endl;
            for (int i = 0; i < ndir; i++)
                std::cout << dir[i] << std::endl;
        }
        mxFree(dir);

        /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
        if (matClose(pmat) != 0)
            throw std::runtime_error("Error closing file '" + filename + "'.");
        pmat = matOpen(filename.c_str(), "r");
        if (pmat == nullptr)
            throw std::runtime_error("Error opening file '" + filename + "'.");

        /* Get headers of all variables */
        std::cout << std::endl << "Examining the header for each variable:" << std::endl;
        for (const std::string &name : array_names) {
            mxArray *pa = matGetVariable(pmat, name.c_str());
            if (pa == nullptr)
                throw std::runtime_error("Error reading in file '" + filename + "'.");
            /* Diagnose header pa */
            mwSize num_of_dimensions = mxGetNumberOfDimensions(pa);
            std::cout << "According to its header, array '" << name << "' has " << num_of_dimensions << " dimensions" << std::endl;
            if (num_of_dimensions != 2)
                throw std::runtime_error("Can only handle arrays with a dimension of 2.");
            const double *data_ptr = mxGetPr(pa);
            MatrixType matrix(mxGetM(pa), mxGetN(pa));
            for (int row = 0; row < matrix.rows(); row++)
            for (int col = 0; col < matrix.cols(); col++)
                matrix(row, col) = data_ptr[row + col * matrix.rows()];
            mxDestroyArray(pa);
            array_map[name] = std::move(matrix);
        }

        if (matClose(pmat) != 0)
            throw std::runtime_error("Error closing file '" + filename + "'.");

        return array_map;
    }

    //int main() {
    //	MATFile *pmat;
    //	mxArray *pa1, *pa2, *pa3;
    //	double data[9] = { 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0 };
    //	const char *file = "mattest.mat";
    //	char str[BUFSIZE];
    //	int status;
    //
    //	printf("Creating file %s...\n\n", file);
    //	pmat = matOpen(file.c_str(), "w");
    //	if (pmat == NULL) {
    //		printf("Error creating file %s\n", file);
    //		printf("(Do you have write permission in this directory?)\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	pa1 = mxCreateDoubleMatrix(3, 3, mxREAL);
    //	if (pa1 == NULL) {
    //		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
    //		printf("Unable to create mxArray.\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	pa2 = mxCreateDoubleMatrix(3, 3, mxREAL);
    //	if (pa2 == NULL) {
    //		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
    //		printf("Unable to create mxArray.\n");
    //		return(EXIT_FAILURE);
    //	}
    //	memcpy((void *)(mxGetPr(pa2)), (void *)data, sizeof(data));
    //
    //	pa3 = mxCreateString("MATLAB: the language of technical computing");
    //	if (pa3 == NULL) {
    //		printf("%s :  Out of memory on line %d\n", __FILE__, __LINE__);
    //		printf("Unable to create string mxArray.\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	status = matPutVariable(pmat, "LocalDouble", pa1);
    //	if (status != 0) {
    //		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
    //		return(EXIT_FAILURE);
    //	}
    //
    //	status = matPutVariableAsGlobal(pmat, "GlobalDouble", pa2);
    //	if (status != 0) {
    //		printf("Error using matPutVariableAsGlobal\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	status = matPutVariable(pmat, "LocalString", pa3);
    //	if (status != 0) {
    //		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
    //		return(EXIT_FAILURE);
    //	}
    //
    //	/*
    //	* Ooops! we need to copy data before writing the array.  (Well,
    //	* ok, this was really intentional.) This demonstrates that
    //	* matPutVariable will overwrite an existing array in a MAT-file.
    //	*/
    //	memcpy((void *)(mxGetPr(pa1)), (void *)data, sizeof(data));
    //	status = matPutVariable(pmat, "LocalDouble", pa1);
    //	if (status != 0) {
    //		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
    //		return(EXIT_FAILURE);
    //	}
    //
    //	/* clean up */
    //	mxDestroyArray(pa1);
    //	mxDestroyArray(pa2);
    //	mxDestroyArray(pa3);
    //
    //	if (matClose(pmat) != 0) {
    //		printf("Error closing file %s\n", file);
    //		return(EXIT_FAILURE);
    //	}
    //
    //	/*
    //	* Re-open file and verify its contents with matGetVariable
    //	*/
    //	pmat = matOpen(file, "r");
    //	if (pmat == NULL) {
    //		printf("Error reopening file %s\n", file);
    //		return(EXIT_FAILURE);
    //	}
    //
    //	/*
    //	* Read in each array we just wrote
    //	*/
    //	pa1 = matGetVariable(pmat, "LocalDouble");
    //	if (pa1 == NULL) {
    //		printf("Error reading existing matrix LocalDouble\n");
    //		return(EXIT_FAILURE);
    //	}
    //	if (mxGetNumberOfDimensions(pa1) != 2) {
    //		printf("Error saving matrix: result does not have two dimensions\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	pa2 = matGetVariable(pmat, "GlobalDouble");
    //	if (pa2 == NULL) {
    //		printf("Error reading existing matrix GlobalDouble\n");
    //		return(EXIT_FAILURE);
    //	}
    //	if (!(mxIsFromGlobalWS(pa2))) {
    //		printf("Error saving global matrix: result is not global\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	pa3 = matGetVariable(pmat, "LocalString");
    //	if (pa3 == NULL) {
    //		printf("Error reading existing matrix LocalString\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	status = mxGetString(pa3, str, sizeof(str));
    //	if (status != 0) {
    //		printf("Not enough space. String is truncated.");
    //		return(EXIT_FAILURE);
    //	}
    //	if (strcmp(str, "MATLAB: the language of technical computing")) {
    //		printf("Error saving string: result has incorrect contents\n");
    //		return(EXIT_FAILURE);
    //	}
    //
    //	/* clean up before exit */
    //	mxDestroyArray(pa1);
    //	mxDestroyArray(pa2);
    //	mxDestroyArray(pa3);
    //
    //	if (matClose(pmat) != 0) {
    //		printf("Error closing file %s\n", file);
    //		return(EXIT_FAILURE);
    //	}
    //	printf("Done\n");
    //	return(EXIT_SUCCESS);
    //}

}

#endif
