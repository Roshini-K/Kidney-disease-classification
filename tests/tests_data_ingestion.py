"""Tests for the DataIngestion component."""

import os
import pytest
from pathlib import Path
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.entity.config_entity import DataIngestionConfig


@pytest.fixture
def sample_config(tmp_path):
    """Provide a temporary DataIngestionConfig for testing."""
    return DataIngestionConfig(
        root_dir=tmp_path,
        source_URL="https://drive.google.com/file/d/1i-rMAKmDBfEopDFlKKDry46LoZrr8mUz/view?usp=drive_link",
        local_data_file=tmp_path / "data.zip",
        unzip_dir=tmp_path / "unzipped"
    )


def test_data_ingestion_download_and_extract(sample_config, monkeypatch):
    """Test that DataIngestion.download_file and extract_zip_file create expected files."""

    ingestion = DataIngestion(config=sample_config)

    # Monkeypatch gdown.download to avoid network call
    monkeypatch.setattr("gdown.download", lambda url, path: Path(path).write_text("dummy content"))

    # Run download_file
    ingestion.download_file()
    assert sample_config.local_data_file.exists(), "Downloaded file should exist."

    # Create a fake zip file for extraction
    import zipfile
    with zipfile.ZipFile(sample_config.local_data_file, "w") as zipf:
        zipf.writestr("sample.txt", "hello")

    # Run extract
    ingestion.extract_zip_file()
    extracted_file = sample_config.unzip_dir / "sample.txt"
    assert extracted_file.exists(), "File should be extracted from zip."
