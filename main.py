import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
import numpy as np

# Load dataset
data = pd.read_csv('/Users/anshreyas/Documents/University Of Birmingham/Study Material/Sem 2/Intelligent Software Engineering/lab1_dataset/pytorch.csv')  # Adjust the path

# Select relevant columns
data = data[['Repository', 'State', 'Title', 'Body', 'Labels', 'class']]
data = data.dropna()  # Drop rows with missing values

# Combine text-based columns for TF-IDF representation
data['text'] = data['Title'] + " " + data['Body'] + " " + data['Labels']

# Ensure class labels are binary
assert set(data['class'].unique()) == {0, 1}, "Class labels must be 0 or 1"

# Handle class imbalance by oversampling the minority class
majority = data[data['class'] == 0]
minority = data[data['class'] == 1]
minority_oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
data_balanced = pd.concat([majority, minority_oversampled])

# Results storage
precision_scores = []
recall_scores = []
f1_scores = []

# Repeat the process 30 times
for i in range(30):
    # Train-test split
    train_data, test_data = train_test_split(data_balanced, test_size=0.3, stratify=data_balanced['class'], random_state=i)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data['text'])  # Use the combined 'text' field
    y_train = train_data['class']
    X_test = vectorizer.transform(test_data['text'])
    y_test = test_data['class']

    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Print average results
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1 Score:", np.mean(f1_scores))
