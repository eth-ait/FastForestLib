import numpy as np


class ImageDataReader(object):

    @staticmethod
    def read_raw_data_and_labels(matlab_file, data_var_name='data', labels_var_name='labels'):
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
        # TODO: Agree on a data format. We want -1 to represent no class
        labels = np.ascontiguousarray(mat_labels, dtype=np.int64) - 1
        return data, labels

    @staticmethod
    def read_from_matlab_file_with_random_samples(matlab_file, num_of_samples_per_image, data_var_name='data', labels_var_name='labels'):
        data, labels = ImageDataReader.read_raw_data_and_labels(matlab_file, data_var_name, labels_var_name)
        return ImageData.create_with_random_samples(data, labels, num_of_samples_per_image)

    @staticmethod
    def read_from_matlab_file_with_all_samples(matlab_file, data_var_name='data', labels_var_name='labels'):
        data, labels = ImageDataReader.read_raw_data_and_labels(matlab_file, data_var_name, labels_var_name)
        return ImageData.create_with_all_samples(data, labels)

class ImageData(object):

    @staticmethod
    def create_with_all_samples(data, labels):
        image_data = ImageData(data, labels, None)
        image_data._sample_indices = image_data._select_all_samples()
        return image_data

    @staticmethod
    def create_with_random_samples(data, labels, num_of_samples_per_image):
        image_data = ImageData(data, labels, None)
        image_data._sample_indices = image_data._select_random_samples(num_of_samples_per_image)
        return image_data

    def __init__(self, data, labels, sample_indices):
        assert data.shape == labels.shape
        self._data = data
        self._labels = labels
        self._num_of_labels = self.__compute_num_of_labels()
        self._sample_indices = sample_indices

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

    def _select_random_samples(self, num_of_samples_per_image):
        sample_indices = np.empty((self.num_of_images * num_of_samples_per_image,), dtype=np.int)
        assert isinstance(sample_indices, np.ndarray)
        image_stride = self.image_width * self.image_height
        for image_index in xrange(self.num_of_images):
            pixel_indices = np.arange(image_stride)
            image_index_offset = image_index * image_stride
            image_labels = self.flat_labels[image_index_offset:image_index_offset + image_stride]
            foreground_indices = pixel_indices[image_labels >= 0]
            assert len(foreground_indices) > num_of_samples_per_image
            selected_indices = np.random.choice(foreground_indices, size=num_of_samples_per_image, replace=False)
            sample_indices[image_index * num_of_samples_per_image:(image_index + 1) * num_of_samples_per_image] \
                = image_index_offset + selected_indices
        return sample_indices

    def __compute_num_of_labels(self):
        unique_labels = np.unique(self.flat_labels)
        return np.sum(unique_labels >= 0)

    def create_sample_indices(self):
        return np.copy(self._sample_indices)

    @property
    def num_of_images(self):
        return self._data.shape[0]

    @property
    def num_of_labels(self):
        return self._num_of_labels

    @property
    def image_width(self):
        return self._data.shape[1]

    @property
    def image_height(self):
        return self._data.shape[2]

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def flat_data(self):
        return self._data.reshape((self._data.shape[0] * self._data.shape[1] * self._data.shape[2]))

    @property
    def flat_labels(self):
        return self._labels.reshape((self._labels.shape[0] * self._labels.shape[1] * self._labels.shape[2]))
