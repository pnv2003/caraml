import numpy as np
import pandas as pd
from models.bayes import NaiveBayes

data = pd.read_csv('data/sport.csv')
print("Data:\n", data)
print()

labels = list(data.columns[1:])
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
print("Labels:", labels)
print("X:\n", X)
print("y:", y)
print()

model = NaiveBayes()
model.fit(X, y)
print("Classes:", model.classes)

X_test = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Warm', 'Low', 'Strong', 'Cool', 'Same']
])
print("X_test:\n", X_test)
print("Predictions:", model.predict(X_test))
