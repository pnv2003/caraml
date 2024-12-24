import numpy as np
from reduce.lda import LDA

def test():
    lda = LDA(1)

    X = np.array([
        [1, 2],
        [1.5, 3],
        [3, 2.5],
        [2, 1.5],
        [3, 1],
        [3.5, 2]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    lda.fit(X, y)

    print("Eigenvectors:")
    print(lda.eigenvectors)

    X_transformed = lda.transform(X)

    print("Transformed data:")
    print(X_transformed)