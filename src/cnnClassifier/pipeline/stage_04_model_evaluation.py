"""Stage 04: Evaluation stage for the CNN classifier."""

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger



STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    """Pipeline stage for evaluating the trained CNN model."""

    def __init__(self):
        """Initialize the evaluation pipeline."""
        pass

    def main(self):
        """
        Run the evaluation stage:
        - Load evaluation configuration
        - Evaluate the trained model
        - Save evaluation scores
        - Log results and model into MLflow
        """
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow() #to run ml-flow




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            