import os
from pathlib import Path
import joblib
import numpy as np
import pytest
from sentiment_model_training.modeling.evaluate import evaluate_model
from sentiment_model_training.modeling.get_data import get_data
import sentiment_model_training.modeling.preprocess as preprocess
from sentiment_model_training.modeling.train import train_model
import warnings
from sklearn.exceptions import UndefinedMetricWarning


@pytest.fixture
def model_train():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    raw_path = Path("data/raw/raw.tsv")
    
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    get_data(url=URL, save_path=str(raw_path))
    
    preprocess.main("data/raw", "data/processed", "model/", max_features=1420)
    
    train_model("data/processed/", "model/")

    yield

    cleanup_files = [raw_path, Path("data/processed/processed.npy"), Path("data/processed/labels.pkl"), Path("data/processed/X_train.pkl"), Path("data/processed/X_test.pkl"), Path("data/processed/y_train.pkl"), Path("data/processed/y_test.pkl"), Path("model/bag_of_words.pkl"), Path("model/model.pkl")]

    for file in cleanup_files:
        if file.exists():
            file.unlink()
        
def test_synonyms(model_train):
    X_test = ["The food and atmosphere were amazing. I love this restaurant!", "The food and atmosphere were amazing. I like this restaurant!", "The food was terrible.", "The food was horrible."]
    y_test = [1, 1, 0, 0]
    
    with open("model/bag_of_words.pkl", "rb") as f:
        bow = joblib.load(f)
    
    X_test_processed = bow.transform(X_test).toarray()
    
    joblib.dump(X_test_processed, os.path.join("data/processed/", "X_test.pkl"))
    joblib.dump(np.array(y_test), os.path.join("data/processed/", "y_test.pkl"))
    
    metrics = evaluate_model(processed_data_path="data/processed", model_path="model/")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        metrics = evaluate_model(processed_data_path="data/processed", model_path="model/")
   
    assert metrics["accuracy"] >= 0.5, "Model accuracy is below expected threshold with synonyms"