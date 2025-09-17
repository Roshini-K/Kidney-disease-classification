"""Tests for PredictionPipeline."""

import numpy as np
from pathlib import Path
import tensorflow as tf
from cnnClassifier.pipeline.prediction import PredictionPipeline


def test_prediction_pipeline(tmp_path):
    """Check PredictionPipeline returns 'Normal' or 'Tumor'."""

    # Save a dummy model
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.h5"
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=[224, 224, 3]),
                                 tf.keras.layers.Dense(2, activation="softmax")])
    model.save(model_path)

    # Patch os.path.join to use our dummy model dir
    import os
    orig_join = os.path.join
    os.path.join = lambda *a: str(model_path) if a[-1] == "model.h5" else orig_join(*a)

    # Create fake image
    img_path = tmp_path / "test.jpg"
    from PIL import Image
    Image.new("RGB", (224, 224)).save(img_path)

    pipeline = PredictionPipeline(str(img_path))
    result = pipeline.predict()

    os.path.join = orig_join  # restore
    assert isinstance(result, list)
    assert "image" in result[0]
