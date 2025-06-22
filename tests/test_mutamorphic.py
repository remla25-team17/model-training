from pathlib import Path
import joblib
import pytest
from lib_ml.preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_model_training.modeling.get_data import get_data
import sentiment_model_training.modeling.preprocess as preprocess
from sentiment_model_training.modeling.train import train_model
import warnings
from sklearn.exceptions import UndefinedMetricWarning


@pytest.fixture
def model_train():
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    raw_path = Path("data/raw/raw.tsv")
    raw_directory = Path("data/raw")
    
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    get_data(url=URL, save_path=str(raw_directory))
    
    preprocess.main("data/raw", "data/processed", "model/", max_features=1420)
    
    train_model("data/processed/", "model/")

    yield

    cleanup_files = [raw_path, Path("data/processed/processed.npy"), Path("data/processed/labels.pkl"), Path("data/processed/X_train.pkl"), Path("data/processed/X_test.pkl"), Path("data/processed/y_train.pkl"), Path("data/processed/y_test.pkl"), Path("model/bag_of_words.pkl"), Path("model/model.pkl")]

    for file in cleanup_files:
        if file.exists():
            file.unlink()
        
def test_mutamorphic_with_automatic_repair(model_train):
    
    with open("model/model.pkl", "rb") as f:
        model = joblib.load(f)

    with open("model/bag_of_words.pkl", "rb") as f:
        bow = joblib.load(f)
    
    test_cases = [
        {
            "original": "The food and atmosphere were amazing. I love this restaurant!",
            "mutants": [
                "The food and atmosphere were amazing. I like this restaurant!",
                "The food and atmosphere were amazing. I enjoy this restaurant!",
                "The food and atmosphere were amazing. I appreciate this restaurant!"
            ]
        },
        {
            "original": "The food was terrible.",
            "mutants": [
                "The food was horrible.",
                "The food was dreadful.",
                "The food was awful."
            ]
        }
    ]

    for case in test_cases:

        corpus = preprocess_text(case["original"])
        numerical = bow.transform([corpus]).toarray()
        pred_orig = model.predict(numerical).tolist()[0]

        repaired = False
        for mutant in case["mutants"]:
            corpus_mut = preprocess_text(mutant)
            x_mut = bow.transform([corpus_mut]).toarray()
            pred_mut = model.predict(x_mut).tolist()[0]

            if pred_mut == pred_orig:
                repaired = True
                print(f"[Repair Successful] '{mutant}' prediction matches original.")
                break  

        assert repaired, (
            f"Mutamorphic inconsistency could not be repaired.\n"
            f"Original: '{case['original']}' → {pred_orig}\n"
            f"Tried mutants: {case['mutants']}"
        )
