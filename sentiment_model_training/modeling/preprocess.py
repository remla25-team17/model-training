"""
This module contains the code for preprocessing the dataset.
"""

import os
import pickle
import pandas as pd
import numpy as np
from lib_ml.preprocessing import embed_reviews


def read_data(raw_dataset_path: str):
    """
    This function reads a raw dataset from a specified path, processes it,
    and saves the processed dataset to another path.

    Parameters:
    - raw_dataset_path (str): The path to the raw dataset.
    """

    # Read the raw dataset
    dataset = pd.read_csv(raw_dataset_path, delimiter='\t', quoting=3)

    return dataset


def preprocess(dataset: pd.DataFrame, max_features: int):
    """
    This function preprocesses the dataset by embedding the text data.

    Parameters:
    - dataset (pd.DataFrame): The dataset to be preprocessed.
    """

    # Get corpus from the dataset
    embeddings = embed_reviews(dataset)

    X = np.array(embeddings)
    assert X.shape[1] == max_features, "Number of features does not match max_features"
    y = dataset.iloc[:, -1].values
    return X, y


def main(raw_data_path: str, processed_data_path: str, model_path: str, max_features: int):
    """
    Main function to execute the data reading and processing.

    Parameters:
    - raw_data_path (str): The path to the raw dataset directory.
    - processed_data_path (str): The path to the processed dataset directory.
    - model_path (str): The path to the model directory.
    """

    # Create directories if they do not exist
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Set the paths for the raw dataset, processed dataset, and labels
    raw_dataset_path = os.path.join(raw_data_path, "raw.tsv")
    processed_dataset_path = os.path.join(processed_data_path, "processed.npy")
    labels_path = os.path.join(processed_data_path, "labels.pkl")

    dataset = read_data(raw_dataset_path)
    X, y = preprocess(dataset, max_features)

    # Save the processed data
    np.save(processed_dataset_path, X)
    with open(labels_path, "wb") as file:
        pickle.dump(y, file)


if __name__ == "__main__":

    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    MODEL_PATH = "model/"
    MAX_FEATURES = 384

    main(RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, MAX_FEATURES)
