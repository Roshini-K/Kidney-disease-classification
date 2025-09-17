"""Stage 03: Model training for the CNN classifier."""

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger



STAGE_NAME = "Training"



class ModelTrainingPipeline:
    """Pipeline stage for training the CNN model."""

    def __init__(self):
        """Initialize the model training pipeline."""
        pass

    def main(self):
        """
        Run the training stage:
        - Load configuration
        - Initialize training component
        - Load base model
        - Prepare data generators
        - Train and save the model
        """
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
