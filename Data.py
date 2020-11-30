import numpy as np
import pandas as pd
import os
import logging
import itertools
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def input_dataset(input_json_file):
    df = pd.read_json(open(input_json_file),"r",encoding="utf8", lines=True)
    logging.info(df.shape)
    print(df.shape)
    logging.info(df.head())
    print(df.head())
    labels = df.is_sarcastic
    labels.head()
    print(labels.head())

    # Split into train and test data
    x_train, x_test, y_train, y_test = train_test_split(df['headline'], labels, test_size=0.20, random_state=1)
    logging.info(x_train)
    print(x_train)
    logging.info(x_test)
    print(x_test)
    logging.info(y_train)
    print(y_train)
    logging.info(y_test)
    print(y_test)

    #intialiazing tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    # DataFlair - Fit and transform train set, transform test set
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # DataFlair - Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    # DataFlair - Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    print(confusion_matrix(y_test, y_pred, labels=[0,1]))

    return df

if __name__=="__main__":
    input_json_file = "fake_news.json"
    df = input_dataset(input_json_file)