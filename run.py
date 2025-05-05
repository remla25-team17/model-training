import pandas as pd
import pickle

from lib_ml.preprocessing import preprocess_reviews

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pickle
from sklearn.naive_bayes import GaussianNB

import joblib

#Importing dataset

dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)

#Data transformation

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)

corpus = preprocess_reviews(dataset)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Saving BoW dictionary to later use in prediction

bow_path = 'bag_of_words.pkl'
pickle.dump(cv, open(bow_path, "wb"))

#Dividing dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Model fitting (Naive Bayes)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'model.pkl') 

