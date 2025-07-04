
import os
from pathlib import Path
import time
import tracemalloc
import joblib
import pytest

from sentiment_model_training.modeling.evaluate import evaluate_model
import sentiment_model_training.modeling.preprocess as preprocess
from sentiment_model_training.modeling.get_data import get_data
from sentiment_model_training.modeling.train import train_model
from sklearn.utils import shuffle


@pytest.fixture()
def dataset():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    raw_path = Path("data/raw/raw.tsv")
    raw_directory = Path("data/raw")

    get_data(url=URL, save_path=str(raw_directory))
    
    yield
    
    cleanup_files = [raw_path, Path("data/processed/processed.npy"), Path("data/processed/labels.pkl"), Path("data/processed/X_train.pkl"), Path("data/processed/X_test.pkl"), Path("data/processed/y_train.pkl"), Path("data/processed/y_test.pkl"), Path("model/bag_of_words.pkl"), Path("model/model.pkl")]

    for file in cleanup_files:
        if file.exists():
            file.unlink()

def test_latency(dataset):
    preprocess.main("data/raw/", "data/processed/", "model/", max_features=1420)
    
    train_model("data/processed/", "model/")
    
    tracemalloc.start() 
    for it in range(100):
        X_test = joblib.load(os.path.join("data/processed/", "X_test.pkl"))
        y_test = joblib.load(os.path.join("data/processed/", "y_test.pkl"))

        X_test, y_test = shuffle(X_test, y_test)

        joblib.dump(X_test, os.path.join("data/processed/", "X_test.pkl"))
        joblib.dump(y_test, os.path.join("data/processed/", "y_test.pkl"))
        start_time = time.time()
        evaluate_model(processed_data_path="data/processed", model_path="model/")
        latency = time.time() - start_time
        
        assert latency < 1, f"Latency exceeded 1 second: {latency:.2f} seconds"
       
    
def test_memory_usage(dataset):
    preprocess.main("data/raw/", "data/processed/", "model/", max_features=1420)
    
    train_model("data/processed/", "model/")
    
    tracemalloc.start() 
    for it in range(100):
        X_test = joblib.load(os.path.join("data/processed/", "X_test.pkl"))
        y_test = joblib.load(os.path.join("data/processed/", "y_test.pkl"))

        X_test, y_test = shuffle(X_test, y_test)

        joblib.dump(X_test, os.path.join("data/processed/", "X_test.pkl"))
        joblib.dump(y_test, os.path.join("data/processed/", "y_test.pkl"))
        evaluate_model(processed_data_path="data/processed", model_path="model/")
       
    current, peak = tracemalloc.get_traced_memory()
    peak = peak / (1024 * 1024) 
    tracemalloc.stop() 
    
    assert peak < 50, "Peak memory usage exceeded 50 MB"