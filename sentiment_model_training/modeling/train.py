"""
This module contains the code for training the sentiment model.
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def train_model(processed_data_path: str = "data/processed/",
                model_path: str = "model/", test_size: float = 0.20, random_state: int = 0):
    """
    Main function to execute the model training process.

    Parameters:
    - processed_data_path (str): The path to the processed dataset directory.
    - model_path (str): The path to the model directory.
    """

    X = np.load(os.path.join(processed_data_path, "processed.npy"))
    y = joblib.load(os.path.join(processed_data_path, "labels.pkl"))

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=test_size, random_state=random_state)
    )

    # Model fitting (Naive Bayes)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, os.path.join(model_path, "model.pkl"))

    # Save data for evaluation
    joblib.dump(X_test, os.path.join(processed_data_path, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(processed_data_path, "y_test.pkl"))


if __name__ == "__main__":

    # Set the paths for the dataset and model directories
    PROCESSED_DATA_PATH = "data/processed/"
    MODEL_PATH = "model/"

    train_model(processed_data_path=PROCESSED_DATA_PATH, model_path=MODEL_PATH)
