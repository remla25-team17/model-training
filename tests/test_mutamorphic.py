import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sentiment_model_training.modeling.evaluate import evaluate_model
from sentiment_model_training.modeling.get_data import get_data
import sentiment_model_training.modeling.preprocess as preprocess
from sentiment_model_training.modeling.train import train_model


@pytest.fixture
def model_train():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    save_path = "data/raw/raw.tsv"

    get_data(url=URL, save_path=save_path)
    
    preprocess.main("data/raw", "data/processed/", "model/", max_features=1420)
    
    train_model("data/processed/", "model/")
    
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
        
def test_synonyms(model_train):
    X_test = ["The food and atmosphere were amazing. I love this restaurant!", "The food and atmosphere were amazing. I like this restaurant!", "The food was terrible.", "The food was horrible."]
    y_test = [1, 1, 0, 0]
    
    with open("model/bag_of_words.pkl", "rb") as f:
        bow = joblib.load(f)
    
    X_test_processed = bow.transform(X_test).toarray()
    
    joblib.dump(X_test_processed, os.path.join("data/processed/", "X_test.pkl"))
    joblib.dump(np.array(y_test), os.path.join("data/processed/", "y_test.pkl"))
    
    metrics = evaluate_model(processed_data_path="data/processed", model_path="model/")
   
    assert metrics["accuracy"] >= 0.5, "Model accuracy is below expected threshold with synonyms"