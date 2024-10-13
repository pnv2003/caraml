import numpy as np

def is_numeric(X):
    return np.issubdtype(X.dtype, np.number)