import numpy as np
from utils import euclidean_distance

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def _predict_item(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: self.k]
        k_labels = [self.y_train[i] for i in k_idx]
        most_common = max(set(k_labels), key=k_labels.count)
        return most_common

    def predict(self, X):
        return np.array([self._predict_item(x) for x in X])
