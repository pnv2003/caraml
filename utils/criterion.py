import math

def entropy(y):

    return -sum(
        p * math.log(p)
        for p in (
            y.count(c) / len(y)
            for c in set(y)
        )
    )

def gain_ratio(ys):

    def info(ys):
        return sum(
            entropy(y) * len(y) / len(ys)
            for y in ys
        )
    
    def gain(ys):
        return entropy(ys) - info(ys)
    
    def split_info(ys):
        return -sum(
            len(y) / len(ys) * math.log(len(y) / len(ys))
            for y in ys
        )
    
    return gain(ys) / split_info(ys) if split_info(ys) != 0 else 0

def gini_index(ys):

    def gini(y):
        return 1 - sum(
            (y.count(c) / len(y)) ** 2
            for c in set(y)
        )
    
    return sum(
        gini(y) * len(y) / len(ys)
        for y in ys
    )