import numpy as np
from math import log


class HistogramStatistics:

    def __init__(self, histogram):
        if isinstance(histogram, HistogramStatistics):
            self._histogram = histogram._histogram
            self._num_of_samples = histogram._num_of_samples
        elif isinstance(histogram, np.ndarray):
            self._histogram = histogram
            self._num_of_samples = np.sum(histogram)
        else:
            raise TypeError("Argument should be a numpy array or a HistogramStatistics object")

    @property
    def num_of_samples(self):
        return self._num_of_samples

    def entropy(self):
        ent = 0
        for i in xrange(len(self._histogram)):
            count = self._histogram[i]
            ent -= count * log(count, 2) / self._num_of_samples
        return ent


class ImageData:

    def __init__(self, data, labels, num_of_samples_per_image):
        assert data.shape == labels.shape
        self._data = data
        self._labels = labels
        self._sample_indices = self._select_random_samples(num_of_samples_per_image)
        pass

    def _select_random_samples(self, num_of_samples_per_image):
        sample_indices = np.empty((self.num_of_images * num_of_samples_per_image,), dtype=np.int)
        assert isinstance(sample_indices, np.ndarray)
        for image_index in xrange(self.num_of_images):
            pixel_indices = np.arange(self.image_width * self.image_height)
            foreground_indices = pixel_indices[self.flat_labels >= 0]
            assert len(foreground_indices) > num_of_samples_per_image
            selected_indices = np.random.choice(foreground_indices, size=num_of_samples_per_image, replace=False)
            index_offset = image_index * self.image_width * self.image_height
            sample_indices[index_offset:index_offset + num_of_samples_per_image] = selected_indices
        return sample_indices

    @property
    def sample_indices(self):
        return self._sample_indices

    @property
    def num_of_images(self):
        return self._data.shape[0]

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
        return self._data.reshape((self._data.shape[0], self._data.shape[1] * self._data.shape[2]))

    @property
    def flat_labels(self):
        return self._labels.reshape((self._labels.shape[0], self._labels.shape[1] * self._labels.shape[2]))


class ImageTrainingContext:

    FEATURE_OFFSET_WINDOW = 20
    THRESHOLD_RANGE_LOW = -1.0
    THRESHOLD_RANGE_HIGH = 1.0

    def __init__(self, image_data):
        self._image_data = image_data
        self._num_of_labels = self._compute_num_of_labels(image_data)
        self._statistics_bins = np.arange(self._num_of_labels)

    def _compute_num_of_labels(self, image_data):
        labels = image_data.flat_labels
        unique_labels = np.unique(labels)
        return len(unique_labels)

    @property
    def sample_indices(self):
        return self._image_data.sample_indices

    @property
    def num_of_labels(self):
        return self._num_of_labels

    def compute_statistics(self, sample_indices):
        labels = self._image_data.flat_labels[sample_indices]
        hist = np.histogram(labels, bins=self._statistics_bins)
        statistics = HistogramStatistics(hist)
        return statistics

    def sample_random_features_and_thresholds(self, num_of_features, num_of_thresholds):
        features = np.random.rand_integers(-self.FEATURE_OFFSET_WINDOW, +self.FEATURE_OFFSET_WINDOW,
                                           size=(num_of_features, 4))
        thresholds = np.random.uniform(self.THRESHOLD_RANGE_LOW, self.THRESHOLD_RANGE_HIGH,
                                       size=(num_of_features, num_of_thresholds))
        return features, thresholds

    def compute_information_gain(self, parent_statistics, left_child_statistics, right_child_statistics):
        parent_statistics = HistogramStatistics(parent_statistics)
        left_child_statistics = HistogramStatistics(left_child_statistics)
        right_child_statistics = HistogramStatistics(right_child_statistics)

        information_gain = parent_statistics.entropy() \
            - (left_child_statistics.num_of_samples * left_child_statistics.entropy()
               + right_child_statistics.num_of_samples * right_child_statistics.entropy()) \
            / parent_statistics.num_of_samples
        return information_gain

    def get_label(self, sample_index):
        return self._image_data.flat_labels[sample_index]

    def compute_feature_value(self, sample_index, feature):
        image_index = sample_index / (self._image_data.image_width * self._image_data.image_height)
        local_index = sample_index % (self._image_data.image_width * self._image_data.image_height)
        local_x = local_index / self._image_data.image_height
        local_y = local_index % self._image_data.image_height
        offset_x1, offset_y1, offset_x2, offset_y2 = feature

        if local_x + offset_x1 < 0 or local_x + offset_x1 >= self._image_data.image_width \
                or local_y + offset_y1 < 0 or local_y + offset_y1 >= self._image_data.image_height:
            pixel1 = 0
        else:
            pixel1 = self._image_data.data[image_index, local_x + offset_x1, local_y + offset_y1]

        if local_x + offset_x2 < 0 or local_x + offset_x2 >= self._image_data.image_width \
                or local_y + offset_y2 < 0 or local_y + offset_y2 >= self._image_data.image_height:
            pixel2 = 0
        else:
            pixel2 = self._image_data.data[image_index, local_x + offset_x2, local_y + offset_y2]

        return pixel1 - pixel2
