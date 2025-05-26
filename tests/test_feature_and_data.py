import os
import pickle
import time
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.proportion import proportion_confint
import numpy as np
import pytest
from sentiment_model_training.modeling.get_data import get_data
from sentiment_model_training.modeling.preprocess import main, read_data

@pytest.fixture
def dataset():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    save_path = "data/raw/raw.tsv"

    get_data(url=URL, save_path=save_path)
    raw_dataset = read_data(save_path)
    
    main("data/raw", "data/processed", "model/", max_features=1420)
    
    processed_dataset = np.load("data/processed/processed.npy")
    
    with open("data/processed/labels.pkl", "rb") as f:
        labels = pickle.load(f)
        
        
    yield raw_dataset, processed_dataset, labels
    
    if os.path.exists("data/raw/raw.tsv"):
        os.remove("data/raw/raw.tsv")
        
    if os.path.exists("data/processed/processed.npy"):
        os.remove("data/processed/processed.npy")
        
    if os.path.exists("data/processed/labels.pkl"):
        os.remove("data/processed/labels.pkl")
        
    if os.path.exists("model/bag_of_words.pkl"):
        os.remove("model/bag_of_words.pkl")
    
def test_raw_data(dataset):
    assert all(data.Review.strip() != "" for data in dataset[0].itertuples()), "Data contains empty reviews"
    assert all(isinstance(data.Review, str) for data in dataset[0].itertuples()), "Data contains non-string reviews"
    assert all(isinstance(data.Liked, int) for data in dataset[0].itertuples()), "Data contains non-integer 'Liked' values"
    assert all(data.Liked in [0, 1] for data in dataset[0].itertuples()), "Data contains invalid 'Liked' values (not 0 or 1)"
    
def test_distribution_of_labels(dataset):
    labels, count = np.unique(dataset[2], return_counts=True)
    total = count.sum()
    
    ci_low_label_negative, ci_upp_label_negative = proportion_confint(count[0], total, alpha=0.05, method='wilson')
    ci_low_label_positive, ci_upp_label_positive = proportion_confint(count[1], total, alpha=0.05, method='wilson')
    
    assert ci_low_label_negative > 0.4 and ci_upp_label_negative < 0.6, "Distribution of label 0 is not within expected bounds (0.40, 0.60)"
    assert ci_low_label_positive > 0.4 and ci_upp_label_positive < 0.6, "Distribution of label 1 is not within expected bounds (0.40, 0.60)"
    
def test_latency_of_feature(dataset):
    max_features = 1420
    for feature in range(max_features):
        classifier = GaussianNB()
        start_time = time.time()
        classifier.fit(dataset[1][:, [feature]], dataset[2])
        latency = time.time() - start_time
        assert latency < 0.5, f"Latency for feature {feature} exceeds 0.5 seconds"


