import os
import pickle
import pandas as pd
import numpy as np
from lib_ml.preprocessing import preprocess_reviews
from sklearn.feature_extraction.text import CountVectorizer

def read_data(raw_dataset_path: str):
    """
    This function reads a raw dataset from a specified path, processes it, and saves the processed dataset to another path.

    Parameters:
    - raw_dataset_path (str): The path to the raw dataset.
    """

    # Read the raw dataset
    dataset = pd.read_csv(raw_dataset_path, delimiter='\t', quoting=3)

    return dataset

def preprocess(dataset: pd.DataFrame, max_features: int = 1420):
    """
    This function preprocesses the dataset by transforming the text data into a bag-of-words representation.

    Parameters:
    - dataset (pd.DataFrame): The dataset to be preprocessed.
    """

    # Get corpus from the dataset
    corpus = preprocess_reviews(dataset)

    # Create a CountVectorizer to convert text data into a bag-of-words representation
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()

    # Extract labels from the dataset
    y = dataset.iloc[:, -1].values

    return X, y, cv

def main(raw_data_path: str, processed_data_path: str, model_path: str, max_features: int = 1420):
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

    # Set the path for the bag-of-words model
    bow_path = os.path.join(model_path, "bag_of_words.pkl")

    dataset = read_data(raw_dataset_path)
    X, y, cv = preprocess(dataset, max_features)

    # Save the processed data
    np.save(processed_dataset_path, X)
    pickle.dump(y, open(labels_path, "wb"))
    pickle.dump(cv, open(bow_path, "wb"))

if __name__ == "__main__":

    raw_data_path = "data/raw"
    processed_data_path = "data/processed"
    model_path = "model/"
    max_features = 1420

    main(raw_data_path, processed_data_path, model_path, max_features)
