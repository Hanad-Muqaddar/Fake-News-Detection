import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import    PassiveAggressiveClassifier
import pickle

if __name__ == '__main__':
    df = pd.read_csv('news.csv')
    labels = df.label
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    tfidfvectriozer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidfvectriozer.fit_transform(x_train)
    tfidf_test = tfidfvectriozer.transform(x_test)
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    file = open('model.pkl', 'wb')
    pickle.dump(pac, file)
    file.close()
