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

        self.maps = None
        self.invmaps = None
    
    def fit(self, X):

        self.maps = []
        self.invmaps = []

        for i in range(X.shape[1]):
            map = {}
            invmap = {}

            dim = len(set(X[:, i]))
            for j, k in enumerate(set(X[:, i])):
                vec = np.zeros(dim)
                vec[j] = 1
                map[k] = vec
                invmap[tuple(vec)] = k

            self.maps.append(map)
            self.invmaps.append(invmap)
            
    def transform(self, X):
        
        res = []
        for i in range(X.shape[1]):
            res.append(np.array([self.maps[i][k] for k in X[:, i]]))
        return np.concatenate(res, axis=1)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):

        res = []
        dims = [len(map) for map in self.maps]

        l, r = None, X
        for i, d in enumerate(dims):
            l, r = r[:, :d], r[:, d:]
            res.append(np.array([
                self.invmaps[i][tuple(v)]
                for v in l
            ]))

        return np.column_stack(res)
    
    def get_encoded_labels(self, labels):

        res = []
        for i, label in enumerate(labels):
            for k, v in self.invmaps[i].items():
                res.append(f"{label}_{v}")            
        return res