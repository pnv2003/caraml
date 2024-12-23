import numpy as np

def rss(y):
    y_mean = np.mean(y)
    return sum((y - y_mean) ** 2)