
import os
from pathlib import Path
import pickle
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sentiment_model_training.modeling.evaluate import evaluate_model
from sentiment_model_training.modeling.get_data import get_data
import sentiment_model_training.modeling.preprocess as preprocess
from sentiment_model_training.modeling.train import train_model
from sentiment_model_training.modeling.preprocess import main, read_data


@pytest.fixture
def dataset():    
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    raw_path = Path("data/raw/raw.tsv")
    
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    get_data(url=URL, save_path=str(raw_path))
    raw_dataset = read_data(str(raw_path))
    main("data/raw", "data/processed", "model/", max_features=1420)

    processed_dataset = np.load(Path("data/processed/processed.npy"))
    with open(Path("data/processed/labels.pkl"), "rb") as f:
        labels = pickle.load(f)

    yield raw_dataset, processed_dataset, labels

    cleanup_files = [raw_path, Path("data/processed/processed.npy"), Path("data/processed/labels.pkl"), Path("data/processed/X_train.pkl"), Path("data/processed/X_test.pkl"), Path("data/processed/y_train.pkl"), Path("data/processed/y_test.pkl"), Path("model/bag_of_words.pkl"), Path("model/model.pkl")]

    for file in cleanup_files:
        if file.exists():
            file.unlink()
        
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

def test_trained_model(model_train):
    initial_metrics = evaluate_model(processed_data_path="data/processed", model_path="model/")
    
    X_train = joblib.load(os.path.join("data/processed/", "X_train.pkl"))
    y_train = joblib.load(os.path.join("data/processed/", "y_train.pkl"))
    X_test = joblib.load(os.path.join("data/processed/", "X_test.pkl"))
    y_test = joblib.load(os.path.join("data/processed/", "y_test.pkl"))
    dummy_model = DummyClassifier(strategy="uniform", random_state=42)
    dummy_model.fit(X_train, y_train)
    y_pred_dummy = dummy_model.predict(X_test)
    dummy_accuracy = accuracy_score(y_test, y_pred_dummy)
    
    assert initial_metrics["accuracy"] > dummy_accuracy, "Trained model accuracy should be greater than dummy model accuracy"
    
def test_model_quality_data_slices(model_train, dataset):
    initial_metrics = evaluate_model(processed_data_path="data/processed", model_path="model/")
    
    preprocessed_data_sliced = []
    labels_sliced = []
    
    preprocessed_data_remained = []
    labels_remained = []
    for index, raw_data in dataset[0].iterrows():
        if len(raw_data["Review"]) < 50:
            preprocessed_data_sliced.append(dataset[1][index])
            labels_sliced.append(dataset[2][index])
        else:
            preprocessed_data_remained.append(dataset[1][index])
            labels_remained.append(dataset[2][index])
            
        
    np.save("data/processed/processed.npy", np.array(preprocessed_data_sliced))
    pickle.dump(np.array(labels_sliced), open("data/processed/labels.pkl", "wb"))
    train_model("data/processed/", "model/")
    metrics_sliced_model = evaluate_model(processed_data_path="data/processed", model_path="model/")
    
    np.save("data/processed/processed.npy", np.array(preprocessed_data_remained))
    pickle.dump(np.array(labels_remained), open("data/processed/labels.pkl", "wb"))
    train_model("data/processed/", "model/")
    metrics_remained_model = evaluate_model(processed_data_path="data/processed", model_path="model/")
    
    
    assert abs(metrics_sliced_model["accuracy"] - initial_metrics["accuracy"]) < 0.1, "Model accuracy should not change significantly when data is sliced"
    assert abs(metrics_remained_model["accuracy"] - initial_metrics["accuracy"]) < 0.1, "Model accuracy should not change significantly when data is sliced"
    
def test_nondeterminism_robustness(model_train):
    initial_metrics = evaluate_model(processed_data_path="data/processed", model_path="model/")
    
    for seed in range(6, 48, 6):
        train_model("data/processed/", "model/")
        metrics = evaluate_model(processed_data_path="data/processed", model_path="model/")
        
        assert abs(metrics["accuracy"] - initial_metrics["accuracy"]) < 0.1, f"Model accuracy should not change significantly with different random states (seed={seed})"
    

    