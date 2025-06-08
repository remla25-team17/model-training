
import os
from pathlib import Path
import subprocess
import pytest

from sentiment_model_training.modeling.get_data import get_data


@pytest.fixture()
def dataset():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    raw_path = Path("data/raw/raw.tsv")
    
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    get_data(url=URL, save_path=str(raw_path))

    yield

    cleanup_files = [raw_path, Path("data/processed/processed.npy"), Path("data/processed/labels.pkl"), Path("data/processed/X_train.pkl"), Path("data/processed/X_test.pkl"), Path("data/processed/y_train.pkl"), Path("data/processed/y_test.pkl"), Path("model/bag_of_words.pkl"), Path("model/model.pkl")]

    for file in cleanup_files:
        if file.exists():
            file.unlink()
            
def test_data_download():
    result = subprocess.run(
        ["python", "sentiment_model_training/modeling/get_data.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0

def test_ML_pipeline(dataset):
    preprocess_step = subprocess.run(
        ["python", "sentiment_model_training/modeling/preprocess.py"],
        check=True,
        capture_output=True,
    )
    
    assert preprocess_step.returncode == 0, "Preprocessing step failed"
    
    training_step = subprocess.run(
        ["python", "sentiment_model_training/modeling/train.py"],
        check=True,
        capture_output=True,
    )
    
    assert training_step.returncode == 0, "Training step failed"
    
    evaluation_step = subprocess.run(
        ["python", "sentiment_model_training/modeling/evaluate.py"],
        check=True,
        capture_output=True,
    )
    
    assert evaluation_step.returncode == 0, "Evaluation step failed"
    assert "accuracy" in evaluation_step.stdout.decode(), "'accuracy' key not found in evaluation output"