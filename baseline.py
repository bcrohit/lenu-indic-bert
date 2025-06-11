"""
Script to train model on Naive Bayes alogorith and use it as our baseline.
"""

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("data\\jur_data_proc.csv")

n_rows = len(data)
n_classes = len(data['label'].unique())

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data['name'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

# Vectorize names as TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 3),  # unigrams + bigrams + trigrams
    token_pattern=r"(?u)\b\w+\b"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression classifier
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_pred, y_test)

print("Logistic Regression accuracy score: ", accuracy)
joblib.dump(clf, 'models\\LG_CLF.joblib')

print("Model saved")