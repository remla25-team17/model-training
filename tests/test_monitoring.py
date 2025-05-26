
import os
import subprocess
import time
import tracemalloc
import joblib
import psutil
import pytest

from sentiment_model_training.modeling.evaluate import evaluate_model
import sentiment_model_training.modeling.preprocess as preprocess
from sentiment_model_training.modeling.get_data import get_data
from sentiment_model_training.modeling.train import train_model
from sklearn.utils import shuffle


@pytest.fixture()
def dataset():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    save_path = "data/raw.tsv"

    get_data(URL=URL, save_path=save_path)
    
    yield
    
    if os.path.exists("data/raw.tsv"):
        os.remove("data/raw.tsv")
        
    if os.path.exists("data/processed.npy"):
        os.remove("data/processed.npy")
        
    if os.path.exists("data/labels.pkl"):
        os.remove("data/labels.pkl")
        
    if os.path.exists("model/bag_of_words.pkl"):
        os.remove("model/bag_of_words.pkl")
        
    if os.path.exists("model/model.pkl"):
        os.remove("model/model.pkl")

def test_latency(dataset):
    preprocess.main("data/", "model/", max_features=1420)
    
    train_model("data/", "model/")
    
    tracemalloc.start() 
    for it in range(100):
        X_test = joblib.load(os.path.join("data/", "X_test.pkl"))
        y_test = joblib.load(os.path.join("data/", "y_test.pkl"))

        X_test, y_test = shuffle(X_test, y_test)

        joblib.dump(X_test, os.path.join("data/", "X_test.pkl"))
        joblib.dump(y_test, os.path.join("data/", "y_test.pkl"))
        start_time = time.time()
        evaluate_model(data_path="data/", model_path="model/")
        latency = time.time() - start_time
        
        assert latency < 1, f"Latency exceeded 1 second: {latency:.2f} seconds"
       
    
def test_memory_usage(dataset):
    preprocess.main("data/", "model/", max_features=1420)
    
    train_model("data/", "model/")
    
    tracemalloc.start() 
    for it in range(100):
        X_test = joblib.load(os.path.join("data/", "X_test.pkl"))
        y_test = joblib.load(os.path.join("data/", "y_test.pkl"))

        X_test, y_test = shuffle(X_test, y_test)

        joblib.dump(X_test, os.path.join("data/", "X_test.pkl"))
        joblib.dump(y_test, os.path.join("data/", "y_test.pkl"))
        evaluate_model(data_path="data/", model_path="model/")
       
    current, peak = tracemalloc.get_traced_memory()
    peak = peak / (1024 * 1024) 
    tracemalloc.stop() 
    
    assert peak < 50, "Peak memory usage exceeded 50 MB"