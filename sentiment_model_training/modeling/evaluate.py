"""
This module contains the code for evaluating the sentiment model.
"""

import argparse
import os
import json
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score
)

def evaluate_model(processed_data_path: str = "data/processed/", model_path: str = "model/"):
    """
    Main function to execute the model evaluation process.

    Parameters:
    - processed_data_path (str): The path to the processed dataset directory.
    - model_path (str): The path to the model directory.
    """

    # Load the test data and labels
    X_test = joblib.load(os.path.join(processed_data_path, "X_test.pkl"))
    y_test = joblib.load(os.path.join(processed_data_path, "y_test.pkl"))

    # Load the trained model
    classifier = joblib.load(os.path.join(model_path, "model.pkl"))

    # Make predictions on the test data
    y_pred = classifier.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")

    # Save to metrics.json
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Generate classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }


if __name__ == "__main__":

    # Set the paths for the dataset and model directories
    PROCESSED_DATA_PATH = "data/processed"
    MODEL_PATH = "model/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=PROCESSED_DATA_PATH,
        help="input path of the data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="path to the model",
    )
    args = parser.parse_args()

    evaluate_model(processed_data_path=args.input, model_path=args.model_path)
