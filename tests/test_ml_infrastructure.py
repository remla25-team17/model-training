
import os
import subprocess
import pytest

from sentiment_model_training.modeling.get_data import get_data


@pytest.fixture()
def dataset():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    save_path = "data/raw/raw.tsv"

    get_data(url=URL, save_path=save_path)
    
    yield
    
    if os.path.exists("data/raw/raw.tsv"):
        os.remove("data/raw/raw.tsv")
        
    if os.path.exists("data/processed/processed.npy"):
        os.remove("data/processed/processed.npy")
        
    if os.path.exists("data/processed/labels.pkl"):
        os.remove("data/processed/labels.pkl")
        
    if os.path.exists("data/processed/X_train.pkl"):
        os.remove("data/processed/X_train.pkl")
        
    if os.path.exists("data/processed/X_test.pkl"):
        os.remove("data/processed/X_test.pkl")
        
    if os.path.exists("data/processed/y_train.pkl"):
        os.remove("data/processed/y_train.pkl")
        
    if os.path.exists("data/processed/y_test.pkl"):
        os.remove("data/processed/y_test.pkl")
        
    if os.path.exists("model/bag_of_words.pkl"):
        os.remove("model/bag_of_words.pkl")
        
    if os.path.exists("model/model.pkl"):
        os.remove("model/model.pkl")

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