import pytest
import os
import pandas as pd



def test_data_balanced():
    os.chdir("../model-training")   
    raw_dataset_path = "./data/raw/historic.tsv"
    dataset = pd.read_csv(raw_dataset_path, delimiter='\t', quoting=3)
    amount_of_positive_reviews = len(dataset[dataset['Liked'] == 1])
    amount_of_negative_reviews = len(dataset[dataset['Liked'] == 0])
    print(amount_of_positive_reviews, amount_of_negative_reviews)
    assert amount_of_positive_reviews / amount_of_negative_reviews > 0.8 # the ratio should be close to 1:1
    assert amount_of_negative_reviews / amount_of_positive_reviews > 0.8


test_data_balanced()