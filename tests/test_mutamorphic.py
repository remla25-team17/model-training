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
    save_path = "data/raw.tsv"

    get_data(URL=URL, save_path=save_path)
    
    preprocess.main("data/", "model/", max_features=1420)
    
    train_model("data/", "model/")
    
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
        
def test_synonyms(model_train):
    X_test = ["The food and atmosphere were amazing. I love this restaurant!", "The food and atmosphere were amazing. I like this restaurant!", "The food was terrible.", "The food was horrible."]
    y_test = [1, 1, 0, 0]
    
    with open("model/bag_of_words.pkl", "rb") as f:
        bow = joblib.load(f)
    
    X_test_processed = bow.transform(X_test).toarray()
    
    joblib.dump(X_test_processed, os.path.join("data/", "X_test.pkl"))
    joblib.dump(np.array(y_test), os.path.join("data/", "y_test.pkl"))
    
    metrics = evaluate_model(data_path="data/", model_path="model/")
   
    assert metrics["accuracy"] >= 0.5, "Model accuracy is below expected threshold with synonyms"
    
def test_synonyms(model_train):
    X_test = ["The food and atmosphere were amazing. I love this restaurant!", "The food and atmosphere were amazing. I like this restaurant!", "The food was terrible.", "The food was horrible."]
    y_test = [1, 1, 0, 0]
    
    with open("model/bag_of_words.pkl", "rb") as f:
        bow = joblib.load(f)
    
    X_test_processed = bow.transform(X_test).toarray()
    
    joblib.dump(X_test_processed, os.path.join("data/", "X_test.pkl"))
    joblib.dump(np.array(y_test), os.path.join("data/", "y_test.pkl"))
    
    metrics = evaluate_model(data_path="data/", model_path="model/")
   
    assert metrics["accuracy"] >= 0.5, "Model accuracy is below expected threshold with synonyms"