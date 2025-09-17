import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """Handles preparation of base CNN model and updates with custom layers."""

    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize PrepareBaseModel class.

        Args:
            config (PrepareBaseModelConfig): Configuration object containing
                model parameters, paths, and hyperparameters.
        """
        self.config = config

    
    def get_base_model(self):
        """
        Fetch the base model (e.g., VGG16) from Keras applications as per config,
        and save it to disk.

        Returns:
            None
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Add custom layers to the base model and configure trainable layers.

        Args:
            model (tf.keras.Model): Base model instance.
            classes (int): Number of output classes.
            freeze_all (bool): Whether to freeze all layers.
            freeze_till (Optional[int]): Number of layers (from the end) to keep trainable.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: Full model ready for training.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        """
        Update the base model by adding custom layers, freezing layers per config,
        and save the updated model.

        Returns:
            None
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save a model to the specified path.

        Args:
            path (Path): File path to save the model.
            model (tf.keras.Model): Keras model to save.

        Returns:
            None
        """
        model.save(path)

