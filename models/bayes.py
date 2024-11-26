from collections import defaultdict
import numpy as np
from models import Model
from utils.common import is_numeric

class NaiveBayes(Model):

    def __init__(self, smoothing=1) -> None:
        super().__init__()
        self.priors = None
        self.likelihoods = None
        self.classes = None
        self.k = smoothing

    def fit(self, X, y):
        self.classes = set(y)
        self.priors = self._calc_prior(y)
        self.likelihoods = self._calc_likelihood(X, y)

    def predict(self, X):
        return np.array([
            self._calc_posterior(X[i]) 
            for i in range(X.shape[0])
        ])

    def _calc_prior(self, y):
        
        # P(c)
        priors = dict()
        for c in self.classes:
            priors[c] = len(y[y == c]) / len(y)
        return priors

    def _calc_likelihood(self, X, y):
        
        # P(i = x | c)
        likelihoods = dict()
        for c in self.classes:
            likelihoods[c] = dict()
            c_index = y == c # rows that have class = c
            for i in range(X.shape[1]):
                if is_numeric(X[:, i]):
                    likelihoods[c][i] = (
                        X[c_index, i].mean(),
                        X[c_index, i].var()
                    )

                else:
                    likelihoods[c][i] = defaultdict(lambda: self._laplace_smoothing(0, len(y[y == c])))
                    for x in set(X[:, i]):
                        x_index = X[:, i] == x # rows that have feature i = x
                        likelihoods[c][i][x] = self._laplace_smoothing(
                            len(y[c_index & x_index]),
                            len(y[c_index])
                        )

        return likelihoods
                

    def _calc_posterior(self, x):
        
        # P(c | x) = P(c) * P(x | c)
        posteriors = {}
        for c in self.classes:
            posterior = self.priors[c]
            for i in range(x.shape[0]):

                if is_numeric(x[i]):
                    mean, var = self.likelihoods[c][i]
                    posterior *= self._gaussian_probability(x[i], mean, var)
                else:
                    posterior *= self.likelihoods[c][i][x[i]]
            posteriors[c] = posterior
        
        return max(posteriors, key=posteriors.get)
    
    def _gaussian_probability(self, x, mean, var):
        
        return (
            1 / np.sqrt(2 * np.pi * var ** 2) *
            np.exp(- (x - mean) ** 2 / (2 * var ** 2))
        )
    
    def _laplace_smoothing(self, count, total):
        return (count + self.k) / (total + self.k * len(self.classes))
        
