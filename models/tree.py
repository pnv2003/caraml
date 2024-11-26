import math
from typing import Literal

import numpy as np
from models import Model
from metrics.regression import rss
from utils.criterion import gain_ratio, gini_index

class Node:

    def __init__(self) -> None:
        self.feat_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.leaf = False

class DecisionTreeClassifier(Model):

    def __init__(self, criterion: Literal['gini', 'entropy'] = 'gini', max_depth=3, min_split=2) -> None:
        
        self.tree = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_split = min_split

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def predict(self, X):
        
        values = []
        
        for x in X:
            node = self.tree
            if node is None:
                raise "not fitted"
            
            while not node.leaf:
                if x[node.feat_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right

            values.append(node.value)

        return np.array(values)
    
    def _grow_tree(self, X, y, depth=0):

        # same class
        if len(set(y)) == 1:
            node = Node()
            node.leaf = True
            node.value = y[0]
            return node
        
        # stop if:
        # - max depth is reached
        # - min split is reached
        if self.max_depth == depth or len(y) < self.min_split:
            node = Node()
            node.leaf = True
            # majority vote
            node.value = max(set(y), key=lambda c: len(y[y == c]))
            return node
        
        # find best split
        X_left, y_left, X_right, y_right, feat_index, threshold = self._best_split(X, y)

        # grow left and right nodes
        node = Node()
        node.feat_index = feat_index
        node.threshold = threshold
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node
        
    def _best_split(self, X, y):
        
        best_X_left = None
        best_y_left = None
        best_X_right = None
        best_y_right = None
        best_index = None
        best_threshold = None
        best_metric = None
        if self.criterion == 'entropy':
            best_metric = -math.inf
        elif self.criterion == 'gini':
            best_metric = math.inf

        for feat in range(X.shape[1]):
                
            thresholds = set(X[:, feat])

            for t in thresholds:

                X_left, y_left, X_right, y_right = self._split(X, y, feat, t)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                if self.criterion == 'entropy':
                    metric = gain_ratio([y_left, y_right])
                    if metric > best_metric:
                        best_metric = metric
                        best_index = feat
                        best_threshold = t
                        best_X_left = X_left
                        best_y_left = y_left
                        best_X_right = X_right
                        best_y_right = y_right

                elif self.criterion == 'gini':
                    metric = gini_index([y_left, y_right])
                    if metric < best_metric:
                        best_metric = metric
                        best_index = feat
                        best_threshold = t
                        best_X_left = X_left
                        best_y_left = y_left
                        best_X_right = X_right
                        best_y_right = y_right
                    
        return best_X_left, best_y_left, best_X_right, best_y_right, best_index, best_threshold

    def _split(self, X, y, feat_index, threshold):
        
        left_idx = X[:, feat_index] < threshold
        right_idx = ~left_idx

        return (
            X[left_idx], y[left_idx],
            X[right_idx], y[right_idx]
        )
    
class DecisionTreeRegressor(Model):
    def __init__(self, max_depth=3, min_split=2) -> None:
        
        self.tree = None
        # self.criterion = criterion
        self.max_depth = max_depth
        self.min_split = min_split

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def predict(self, X):
        
        node = self.tree
        if node is None:
            raise "not fitted"
        
        while not node.leaf:
            if X[node.feat_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value
    
    def _grow_tree(self, X, y, depth=0):

        # same class
        if len(set(y)) == 1:
            node = Node()
            node.leaf = True
            node.value = y[0]
            return node
        
        # stop if:
        # - max depth is reached
        # - min split is reached
        if self.max_depth == depth or len(y) < self.min_split:
            node = Node()
            node.leaf = True
            # mean value
            node.value = np.mean(y)
            return node
        
        # find best split
        X_left, y_left, X_right, y_right, feat_index, threshold = self._best_split(X, y)

        # grow left and right nodes
        node = Node()
        node.feat_index = feat_index
        node.threshold = threshold
        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node
        
    def _best_split(self, X, y):
        
        best_X_left = None
        best_y_left = None
        best_X_right = None
        best_y_right = None
        best_index = None
        best_threshold = None
        best_metric = None
        best_metric = math.inf

        for feat in range(X.shape[1]):
                
            thresholds = set(X[:, feat])

            for t in thresholds:

                X_left, y_left, X_right, y_right = self._split(X, y, feat, t)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                metric = rss(y_left) + rss(y_right)
                if metric < best_metric:
                    best_metric = metric
                    best_X_left = X_left
                    best_y_left = y_left
                    best_X_right = X_right
                    best_y_right = y_right
                    best_index = feat
                    best_threshold = t

        return best_X_left, best_y_left, best_X_right, best_y_right, best_index, best_threshold

    def _split(self, X, y, feat_index, threshold):
        
        left_idx = X[:, feat_index] < threshold
        right_idx = ~left_idx

        return (
            X[left_idx], y[left_idx],
            X[right_idx], y[right_idx]
        )
