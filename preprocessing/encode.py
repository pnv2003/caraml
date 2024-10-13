import numpy as np

class LabelEncoder:

    def __init__(self) -> None:
        self.map = None
        self.invmap = None
    
    def fit(self, y):    
        self.map = {}
        self.invmap = {}
        for i, k in enumerate(set(y)):
            self.map[k] = i
            self.invmap[i] = k

    def transform(self, y):
        return np.array([self.map[k] for k in y])
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return np.array([self.invmap[i] for i in y])

class OneHotEncoder:
    
    def __init__(self) -> None:
        self.map = None
        self.invmap = None
    
    def fit(self, X):
        
        self.map = {}
        self.invmap = {}
        dim = len(set(X))
        for i, k in enumerate(dim):
            vec = np.zeros(dim)
            vec[i] = 1
            self.map[k] = vec
            self.invmap[tuple(vec)] = k

    def transform(self, X):
        return np.array([np.array(self.map[k]) for k in X])
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return np.array([self.invmap[tuple(v)] for v in X])