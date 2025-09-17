"""Test cases for the Evaluation component."""

import tensorflow as tf
from pathlib import Path
from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier.entity.config_entity import EvaluationConfig


def test_evaluation(tmp_path, monkeypatch):
    """Check evaluation computes loss/accuracy with a simple dummy model."""

    # Create & compile a dummy model
    model_path = tmp_path / "model.h5"
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[224, 224, 3]),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.save(model_path)

    # Create fake dataset with 2 classes
    from PIL import Image
    for class_name in ["classA", "classB"]:
        class_dir = tmp_path / "data" / class_name
        class_dir.mkdir(parents=True)
        Image.new("RGB", (224, 224)).save(class_dir / "img.jpg")

    # Build config
    config = EvaluationConfig(
        path_of_model=model_path,
        training_data=str(tmp_path / "data"),
        mlflow_uri="file://" + str(tmp_path),
        all_params={"EPOCHS": 1},
        params_image_size=[224, 224, 3],
        params_batch_size=1,
    )

    eval_obj = Evaluation(config)

    # Monkeypatch the generator to bypass validation_split
    def dummy_generator(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        self.valid_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            shuffle=False,
        )

    monkeypatch.setattr(Evaluation, "_valid_generator", dummy_generator)

    # Run evaluation
    eval_obj.evaluation()

    # Assert that we got valid scores
    assert isinstance(eval_obj.score, list)
    assert len(eval_obj.score) == 2  # [loss, accuracy]
