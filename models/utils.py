import numpy as np

def r2_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))