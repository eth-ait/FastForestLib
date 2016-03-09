"""
This file contains configurations for the random forest trainer and the image weak learner.
"""

from forest_trainer import TrainingParameters
from c_image_weak_learner import Parameters
#from image_weak_learner import Parameters

training_data_parameters = {
    'num_of_samples_per_image': 1000,
}

testing_data_parameters = {}

training_parameters = TrainingParameters(
    maximum_depth=18,
    num_of_features=200,
    num_of_thresholds=200,
    num_of_trees=1,
    minimum_num_of_samples=100
)

weak_learner_parameters = Parameters(
    feature_offset_range_low=3,
    feature_offset_range_high=15,
    threshold_range_low=-300.0,
    threshold_range_high=+300.0,
)

testing_parameters = {}
