import numpy as np
from reduce.pca import PCA

def test():
    X = np.array([
        [1, 2],
        [1.5, 3],
        [3, 2.5],
        [2, 1.5],
        [3, 1],
        [3.5, 2]
    ])

    pca = PCA(1)
    pca.fit(X)

    print("Eigenvectors:")
    print(pca.eigenvectors)

    X_transformed = pca.transform(X)

    print("Transformed data:")
    print(X_transformed)