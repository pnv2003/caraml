import numpy as np

class TSNE:
    """t-distributed Stochastic Neighbor Embedding"""

    def __init__(
        self, 
        n_components: int,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding = None

    def fit_transform(self, X):

        # Compute pairwise distances
        D = np.sum((X[:, np.newaxis] - X) ** 2, axis=-1)

        # Initialize embedding
        embedding = np.random.normal(size=(X.shape[0], self.n_components))

        # Compute P-values
        P = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            indices = np.concatenate([np.r_[0:i], np.r_[i+1:X.shape[0]]])
            distances = D[i, indices]
            p = np.exp(-distances / np.percentile(distances, self.perplexity))
            P[i, indices] = p / np.sum(p)

        # Symmetrize P-values
        P = (P + P.T) / (2 * X.shape[0])

        # Initialize Q-values
        Q = np.zeros((X.shape[0], X.shape[0]))

        # Gradient descent
        for _ in range(self.n_iter):
            distances = np.sum((embedding[:, np.newaxis] - embedding) ** 2, axis=-1)
            Q = 1 / (1 + distances)
            Q[np.arange(X.shape[0]), np.arange(X.shape[0])] = 0
            Q /= np.sum(Q)
            grad = 4 * ((P - Q)[:, :, np.newaxis] * (embedding[:, np.newaxis] - embedding)).sum(axis=1)
            embedding -= self.learning_rate * grad

        self.embedding = embedding
        return self.embedding