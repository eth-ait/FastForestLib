import numpy as np


class ImageData(object):
    """
    This class represents labeled image data.

    Labels are assumed to be consecutive integers starting from 0.
    A label of -1 is considered as a marker for invalid data.
    """

    def __init__(self, data, labels):
        """
        Initialize a new L{ImageData} object.

        @param data: A L{numpy} array of the image data with dimensions (numOfImages x width x height)
        @param labels: A L{numpy} array of the pixel labels with dimensions (numOfImages x width x height)
        @return: A new L{ImageData} object
        """
        assert data.ndim == 3
        assert data.shape == labels.shape
        self._data = data
        self._labels = labels
        self._num_of_labels = self.__compute_num_of_labels()

    def __compute_num_of_labels(self):
        unique_labels = np.unique(self.flat_labels)
        return np.sum(unique_labels >= 0)

    @property
    def num_of_images(self):
        """
        The number of images
        """
        return self._data.shape[0]

    @property
    def num_of_labels(self):
        """
        The number of labels in the data
        """
        return self._num_of_labels

    @property
    def image_width(self):
        """
        The width of the images
        """
        return self._data.shape[1]

    @property
    def image_height(self):
        """
        The height of the images
        """
        return self._data.shape[2]

    @property
    def data(self):
        """
        The image data array
        """
        return self._data

    @property
    def labels(self):
        """
        The pixel labels array
        """
        return self._labels

    @property
    def flat_data(self):
        """
        A flat view of the image data array (the memory layout depends on the provided image data array)
        """
        return self._data.reshape((self._data.shape[0] * self._data.shape[1] * self._data.shape[2]))

    @property
    def flat_labels(self):
        """
        A flat view of the pixel labels array (the memory layout depends on the provided pixel labels array)
        """
        return self._labels.reshape((self._labels.shape[0] * self._labels.shape[1] * self._labels.shape[2]))


class ImageSampleData(ImageData):
    """
    This class represents labeled image data with a selection of sample pixels. The sample pixels can be a subset of
    all the image pixels or they can consist of all the image pixels.
    """

    def __init__(self, data, labels, sample_indices):
        """
        Initialize a new L{ImageSampleData} object.

        @param data: A C-contiguous L{numpy} array of the image data with dimensions (numOfImages x width x height)
        @param labels: A C-contiguous L{numpy} array of the pixel labels with dimensions (numOfImages x width x height)
        @param sample_indices: An array of the pixel indices that are used as samples (the indices should be indices
                               of the flattened image data and pixel labels arrays)
        @return: A new L{ImageSampleData} object
        """
        assert(isinstance(data, np.ndarray))
        assert(isinstance(labels, np.ndarray))
        assert(data.flags.c_contiguous)
        assert(labels.flags.c_contiguous)
        super(ImageSampleData, self).__init__(data, labels)
        self._sample_indices = sample_indices

    @staticmethod
    def create_with_all_samples(data, labels):
        """
        Create a new L{ImageSampleData} object with all the pixels selected as samples.

        @param data: A C-contiguous L{numpy} array of the image data with dimensions (numOfImages x width x height)
        @param labels: A C-contiguous L{numpy} array of the pixel labels with dimensions (numOfImages x width x height)
        @return: A new L{ImageSampleData} object
        """
        image_data = ImageSampleData(data, labels, None)
        image_data._sample_indices = image_data._select_all_samples()
        return image_data

    @staticmethod
    def create_with_random_samples(data, labels, num_of_samples_per_image, \
                                   enforce_num_of_samples_per_image=False):
        """
        Create a new L{ImageSampleData} object with a random number of pixels from each image selected as samples.

        @param data: A C-contiguous L{numpy} array of the image data with dimensions (numOfImages x width x height)
        @param labels: A C-contiguous L{numpy} array of the pixel labels with dimensions (numOfImages x width x height)
        @param num_of_samples_per_image: The number of samples that will be selected from each image.
        @param enforce_num_of_samples_per_image: A flag indicating whether to fail if an image has less than L{num_of_samples_per_image} foreground pixels
        @return: A new L{ImageSampleData} object
        """
        image_data = ImageSampleData(data, labels, None)
        image_data._sample_indices = image_data._select_random_samples(num_of_samples_per_image, enforce_num_of_samples_per_image)
        return image_data

    def _select_all_samples(self):
        sample_indices_list = []
        image_stride = self.image_width * self.image_height
        for image_index in xrange(self.num_of_images):
            pixel_indices = np.arange(image_stride)
            image_index_offset = image_index * image_stride
            image_labels = self.flat_labels[image_index_offset:image_index_offset + image_stride]
            foreground_indices = pixel_indices[image_labels >= 0]
            sample_indices_list.append(image_index_offset + foreground_indices)
        sample_indices = np.hstack(sample_indices_list)
        return sample_indices

    def _select_random_samples(self, num_of_samples_per_image, enforce_num_of_samples_per_image=False):
        sample_indices = np.empty((self.num_of_images * num_of_samples_per_image,), dtype=np.int)
        assert isinstance(sample_indices, np.ndarray)
        image_stride = self.image_width * self.image_height
        sample_index_offset = 0
        for image_index in xrange(self.num_of_images):
            pixel_indices = np.arange(image_stride)
            image_index_offset = image_index * image_stride
            image_labels = self.flat_labels[image_index_offset:image_index_offset + image_stride]
            foreground_indices = pixel_indices[image_labels >= 0]
            if enforce_num_of_samples_per_image:
                assert len(foreground_indices) > num_of_samples_per_image
            if len(foreground_indices) < num_of_samples_per_image:
                selected_indices = foreground_indices
            else:
                selected_indices = np.random.choice(foreground_indices, size=num_of_samples_per_image, replace=False)
            sample_indices[sample_index_offset:(sample_index_offset + len(selected_indices))] \
                = image_index_offset + selected_indices
            sample_index_offset += len(selected_indices)
        sample_indices = sample_indices[:sample_index_offset]
        return sample_indices

    def create_sample_indices(self):
        """
        @return: A L{numpy} array of the selected sample indices
                 (the indices refer to the flat image data and pixel labels arrays)
        """
        return np.copy(self._sample_indices)


class ImageDataReader(object):
    """
    This class provides functions for reading image data (i.e. L{ImageData} or L{ImageSampleData} from MATLAB .mat files
    (the HDF5 format for large files is also supported).

    The saved files are assumed to contain MATLAB arrays of dimension (numOfImages x width x height).
    The actual memory layout is in Fortran-mode for the non-HDF5 MATLAB .mat files,
    so the transpose of the data is taken.
    """

    @staticmethod
    def read_from_matlab_file(matlab_file, data_var_name='data', labels_var_name='labels'):
        """
        Read image data from a MATLAB .mat file.

        @param matlab_file: Filename of the MATLAB .mat file
        @param data_var_name: Variable name of the image data array
        @param labels_var_name: Variable name of the pixel labels array
        @return: A new L{ImageData} object of the corresponding data.
        """
        try:
            import scipy.io
            mat_dict = scipy.io.loadmat(matlab_file, variable_names=(data_var_name, labels_var_name))
            mat_data = mat_dict[data_var_name].T
            mat_labels = mat_dict[labels_var_name].T
        except NotImplementedError, e:
            import h5py
            f = h5py.File(matlab_file)
            mat_data = f.get(data_var_name)
            mat_labels = f.get(labels_var_name)
        data = np.ascontiguousarray(mat_data, dtype=np.float64)
        labels = np.ascontiguousarray(mat_labels, dtype=np.int64)
        return ImageData(data, labels)

    @staticmethod
    def read_from_matlab_file_with_random_samples(matlab_file, num_of_samples_per_image, data_var_name='data',
                                                  labels_var_name='labels'):
        """
        Read image data from a MATLAB .mat file and randomly select a number of pixels from each image as samples.

        @param matlab_file: Filename of the MATLAB .mat file
        @param num_of_samples_per_image: The number of samples that will be selected from each image.
        @param data_var_name: Variable name of the image data array
        @param labels_var_name: Variable name of the pixel labels array
        @return: A new L{ImageSampleData} object of the corresponding data.
        """
        image_data = ImageDataReader.read_from_matlab_file(matlab_file, data_var_name, labels_var_name)
        return ImageSampleData.create_with_random_samples(image_data.data, image_data.labels, num_of_samples_per_image)

    @staticmethod
    def read_from_matlab_file_with_all_samples(matlab_file, data_var_name='data', labels_var_name='labels'):
        """
        Read image data from a MATLAB .mat file and select all pixels from each image as samples.

        @param matlab_file: Filename of the MATLAB .mat file
        @param data_var_name: Variable name of the image data array
        @param labels_var_name: Variable name of the pixel labels array
        @return: A new L{ImageSampleData} object of the corresponding data.
        """
        image_data = ImageDataReader.read_from_matlab_file(matlab_file, data_var_name, labels_var_name)
        return ImageSampleData.create_with_all_samples(image_data.data, image_data.labels)
