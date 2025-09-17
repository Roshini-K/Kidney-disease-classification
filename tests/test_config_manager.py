"""Tests for ConfigurationManager."""

from cnnClassifier.config.configuration import ConfigurationManager


def test_config_manager():
    """Ensure ConfigurationManager returns valid configs."""

    manager = ConfigurationManager()
    ingestion = manager.get_data_ingestion_config()
    assert hasattr(ingestion, "source_URL")

    base_model = manager.get_prepare_base_model_config()
    assert base_model.params_image_size

    training = manager.get_training_config()
    assert training.params_epochs

    eval_config = manager.get_evaluation_config()
    assert eval_config.mlflow_uri
