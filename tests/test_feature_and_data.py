import os
from pathlib import Path
import pickle
import time
from unittest import mock
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.proportion import proportion_confint
import numpy as np
import pytest
from sentiment_model_training.modeling.get_data import get_data
from sentiment_model_training.modeling.preprocess import main, read_data
from scipy.stats import pearsonr

@pytest.fixture
def dataset():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    raw_path = Path("data/raw/raw.tsv")
    raw_directory = Path("data/raw")
    processed_dir = Path("data/processed")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    Path("model/bag_of_words.pkl").parent.mkdir(parents=True, exist_ok=True)

    get_data(url=URL, save_path=str(raw_directory))
    raw_dataset = read_data(str(raw_path))
    main("data/raw", "data/processed", "model/", max_features=1420)

    processed_dataset = np.load(Path("data/processed/processed.npy"))
    with open(Path("data/processed/labels.pkl"), "rb") as f:
        labels = pickle.load(f)
        
    with open(Path("model/bag_of_words.pkl"), "rb") as f:
        bag_of_words = pickle.load(f)

    yield raw_dataset, processed_dataset, labels, bag_of_words
    
    cleanup_files = [raw_path, Path("data/processed/processed.npy"), Path("data/processed/labels.pkl"), Path("model/bag_of_words.pkl"), Path("model/bag_of_words.pkl")]

    for file in cleanup_files:
        if file.exists():
            file.unlink()
    
def test_raw_data(dataset):
    assert all(data.Review.strip() != "" for data in dataset[0].itertuples()), "Data contains empty reviews"
    assert all(isinstance(data.Review, str) for data in dataset[0].itertuples()), "Data contains non-string reviews"
    assert all(isinstance(data.Liked, int) for data in dataset[0].itertuples()), "Data contains non-integer 'Liked' values"
    assert all(data.Liked in [0, 1] for data in dataset[0].itertuples()), "Data contains invalid 'Liked' values (not 0 or 1)"
    
# def test_processed_data(dataset):
#     assert dataset[1].shape[0] == dataset[0].shape[0], "Processed data does not match raw data length"
#     assert dataset[1].shape[1] == 1420, "Processed data does not have the expected number of features"
#     assert dataset[2].shape[0] == dataset[0].shape[0], "Labels do not match raw data length"
    
#     assert dataset[3] is not None, "Bag of words model is not loaded"
#     assert hasattr(dataset[3], 'transform'), "Bag of words model does not have a 'transform' method"
#     assert hasattr(dataset[3], 'vocabulary_'), "Bag of words model does not have a 'vocabulary_' attribute"
#     assert len(dataset[3].vocabulary_) > 0, "Bag of words model vocabulary is empty"
    
    
# def test_distribution_of_labels(dataset):
#     _, count = np.unique(dataset[2], return_counts=True)
#     total = count.sum()
    
#     ci_low_label_negative, ci_upp_label_negative = proportion_confint(count[0], total, alpha=0.05, method='wilson')
#     ci_low_label_positive, ci_upp_label_positive = proportion_confint(count[1], total, alpha=0.05, method='wilson')
    
#     assert ci_low_label_negative > 0.4 and ci_upp_label_negative < 0.6, "Distribution of label 0 is not within expected bounds (0.40, 0.60)"
#     assert ci_low_label_positive > 0.4 and ci_upp_label_positive < 0.6, "Distribution of label 1 is not within expected bounds (0.40, 0.60)"
    
# def test_latency_of_feature(dataset):
#     max_features = 1420
#     for feature in range(max_features):
#         classifier = GaussianNB()
#         start_time = time.time()
#         classifier.fit(dataset[1][:, [feature]], dataset[2])
#         latency = time.time() - start_time
#         assert latency < 0.5, f"Latency for feature {feature} exceeds 0.5 seconds"
        
# def test_feature_label_correlation(dataset):
#     features = dataset[1]
#     labels = dataset[2]
#     num_features = features.shape[1]
#     correlations = []
#     for i in range(num_features):
#         corr, _ = pearsonr(features[:, i], labels)
#         correlations.append(abs(corr))
#     mean_correlation = np.mean(correlations)
#     print(mean_correlation)
#     assert mean_correlation > 0.01, "Mean correlation between features and labels is too low"