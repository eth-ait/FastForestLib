from forest_trainer import RandomForestTrainer, TrainingParameters
from image_random_forest import ImageData, ImageTrainingContext


if __name__ == '__main__':

    training_parameters = TrainingParameters()
    data = ImageData()
    sample_indices = data.get_sample_indices()
    training_context = ImageTrainingContext(data)

    trainer = RandomForestTrainer()
    trainer.train_forest(sample_indices, training_context, training_parameters)
