import numpy as np

class PCA:
    """Principal Component Analysis"""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.eigenvectors = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = X.T @ X / X.shape[0]
        eigvals, eigvecs = np.linalg.eig(cov)
        self.eigenvectors = eigvecs[:, np.argsort(eigvals)[::-1][:self.n_components]]

    def transform(self, X):
        return (X - self.mean) @ self.eigenvectors
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


        
