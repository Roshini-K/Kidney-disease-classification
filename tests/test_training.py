"""Test cases for the Training component."""

import tensorflow as tf
from pathlib import Path
from cnnClassifier.components.model_training import Training
from cnnClassifier.entity.config_entity import TrainingConfig


def test_training_pipeline(tmp_path):
    """Check that training produces a saved model."""

    # Create fake training data (tiny dataset with 1 class and 1 image)
    train_dir = tmp_path / "train"
    class_dir = train_dir / "classA"
    class_dir.mkdir(parents=True)

    from PIL import Image
    Image.new("RGB", (224, 224)).save(class_dir / "img.jpg")

    # Build config
    config = TrainingConfig(
        root_dir=tmp_path,
        trained_model_path=tmp_path / "trained.h5",
        updated_base_model_path=tmp_path / "dummy_base.h5",
        training_data=train_dir,
        params_epochs=1,
        params_batch_size=1,
        params_is_augmentation=False,
        params_image_size=[224, 224, 3],
    )

    # Save a compiled dummy base model
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[224, 224, 3]),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    dummy_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    dummy_model.save(config.updated_base_model_path)

    # Run training pipeline
    trainer = Training(config)
    trainer.get_base_model()
    trainer.train_valid_generator()
    trainer.train()

    # Check that trained model file exists
    assert config.trained_model_path.exists()
    # Ensure it's a valid keras model
    loaded = tf.keras.models.load_model(config.trained_model_path)
    assert isinstance(loaded, tf.keras.Model)
