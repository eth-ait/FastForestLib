"""
This file contains configurations for the random forest trainer and the image weak learner.
"""

from forest_trainer import TrainingParameters
from c_image_weak_learner import Parameters

training_data_parameters = {
    'num_of_samples_per_image': 150,
}

testing_data_parameters = {}

training_parameters = TrainingParameters(
    maximum_depth=20,
    num_of_features=100,
    num_of_thresholds=5,
    num_of_trees=1,
    minimum_num_of_samples=100
)

weak_learner_parameters = Parameters(
    feature_offset_range_low=3,
    feature_offset_range_high=15,
    threshold_range_low=-0.5,
    threshold_range_high=+0.5,
)

testing_parameters = {}

