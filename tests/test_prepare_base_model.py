"""Tests for PrepareBaseModel component."""

import tensorflow as tf
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


def test_prepare_base_model(tmp_path):
    """Ensure base model and updated model are created and saved."""

    config = PrepareBaseModelConfig(
        root_dir=tmp_path,
        base_model_path=tmp_path / "base.h5",
        updated_base_model_path=tmp_path / "updated.h5",
        params_image_size=[224, 224, 3],
        params_learning_rate=0.001,
        params_include_top=False,
        params_weights=None,  # keep None for fast testing
        params_classes=2,
    )

    pbm = PrepareBaseModel(config)
    pbm.get_base_model()
    assert config.base_model_path.exists()

    pbm.update_base_model()
    assert config.updated_base_model_path.exists()

    model = tf.keras.models.load_model(config.updated_base_model_path)
    assert isinstance(model, tf.keras.Model)
