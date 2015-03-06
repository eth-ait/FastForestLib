from forest_trainer import TrainingParameters
from c_image_weak_learner import Parameters

training_parameters = TrainingParameters(
    maximum_depth=10,
    num_of_features=10,
    num_of_thresholds=10,
    num_of_trees=1,
    minimum_num_of_samples=100
)

weak_learner_parameters = Parameters(
    feature_offset_range_low=3,
    feature_offset_range_high=15,
    threshold_range_low=-300.0,
    threshold_range_high=+300.0,
)
