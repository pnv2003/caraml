import numpy as np

def split_train_test(data, test_ratio=0.1, seed=None):

    np.random.seed(seed)
    idx = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return data.iloc[train_idx], data.iloc[test_idx]

def split_loocv(data):
    
    n = len(data)
    for i in range(n):
        train = data.drop(i)
        test = data.iloc[i]
        yield train, test

def split_kfold(data, k=5, seed=None):
    
    n = len(data)
    fold_size = n // k
    np.random.seed(seed)
    idx = np.random.permutation(n)
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        test = data.iloc[idx[start:end]]
        train = data.drop(idx[start:end])
        yield train, test
