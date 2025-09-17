import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    """Handles model evaluation and logging of results to MLflow."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the Evaluation class.

        Args:
            config (EvaluationConfig): Configuration object containing
                paths, parameters, and MLflow settings.
        """
        self.config = config

    
    def _valid_generator(self):

        """
        Create and configure the validation data generator.

        Returns:
            None
        """
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Load a TensorFlow Keras model from the specified path.

        Args:
            path (Path): Path to the saved model.

        Returns:
            tf.keras.Model: Loaded model.
        """
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        """
        Evaluate the model on the validation dataset and save scores.

        Returns:
            None
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """
        Save evaluation scores (loss and accuracy) into a JSON file.

        Returns:
            None
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        """
        Log evaluation parameters, metrics, and model into MLflow.

        Notes:
            - Model registry is not supported on all backends (e.g., DagsHub).
            - If using a simple file-based store, model registration is skipped.

        Returns:
            None
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            mlflow.keras.log_model(self.model, "model")
