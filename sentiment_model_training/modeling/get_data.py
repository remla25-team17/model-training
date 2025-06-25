"""
This module contains the code for downloading the dataset from a given URL
 and saving it to a specified path.
"""

import argparse
import os
import requests


def get_data(url: str, save_path: str):
    """
    This function downloads the dataset from a given URL and saves it to a specified path.

    Parameters:
    - URL (str): The URL to download the dataset from.
    save_path (str): The local path where the dataset will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download the dataset
    response = requests.get(url, timeout=5)

    if response.status_code == 200:
        with open(os.path.join(save_path, "raw.tsv"), 'wb') as file:
            file.write(response.content)
        print(f"Dataset downloaded and saved to {save_path}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")


if __name__ == "__main__":

    # Set the URL and path for the dataset
    URL = (
        "https://raw.githubusercontent.com/proksch"
        "/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    )
    SAVE_PATH = "data/raw/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=URL,
        help="url for where to download the raw dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=SAVE_PATH,
        help="directory for where to save the raw dataset",
    )
    args = parser.parse_args()

    get_data(url=args.url, save_path=args.save_path)
