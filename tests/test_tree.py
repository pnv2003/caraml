import numpy as np
import pandas as pd
from model.tree import DecisionTreeClassifier
from preprocess.encoder import OneHotEncoder

def test():
    data = pd.read_csv('data/golf.csv', dtype={'Windy': 'str'})
    print("Data:\n", data)

    labels = list(data.columns[1:])
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    print("Labels:", labels)
    print("X:\n", X)
    print("y:", y)

    enc = OneHotEncoder()
    X = enc.fit_transform(X)
    print("X (encoded):\n", X)

    x_labels = enc.get_encoded_labels(labels[:-1])
    y_label = labels[-1]
    print("X labels:", x_labels)
    print("y label:", y_label)

    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,
        min_split=2
    )
    model.xlabels = x_labels
    model.ylabel = y_label
    model.fit(X, y)

    X_test = enc.transform(
        np.array([
            ['cool', 'sunny', 'normal', 'false'],
            ['mild', 'sunny', 'normal', 'false']
        ])
    )
    print("X_test (encoded):\n", X_test)
    print("Predictions:", model.predict(X_test))
