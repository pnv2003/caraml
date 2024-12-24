import numpy as np

class LDA:
    """Linear Discriminant Analysis"""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.eigenvectors = None

    def fit(self, X, y):
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for i in range(n_classes):
            X_i = X[y == i]
            mean_i = np.mean(X_i, axis=0)
            S_W += (X_i - mean_i).T @ (X_i - mean_i)
            S_B += len(X_i) * np.outer(mean_i - mean_overall, mean_i - mean_overall)

        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        self.eigenvectors = eigvecs[:, np.argsort(eigvals)[::-1][:self.n_components]]

    def transform(self, X):
        return X @ self.eigenvectors
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)