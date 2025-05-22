import os
import requests

def get_data(URL: str, save_path: str):
    """
    This function downloads the dataset from a given URL and saves it to a specified path.

    Parameters:
    - URL (str): The URL to download the dataset from.
    save_path (str): The local path where the dataset will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download the dataset
    response = requests.get(URL)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Dataset downloaded and saved to {save_path}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":

    # Set the URL and path for the dataset
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    save_path = "data/raw.tsv"

    get_data(URL=URL, save_path=save_path)