import numpy as np
from utils import sigmoid

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.cost_history = np.zeros(self.n_iters)

    def cost(self, x, y, m):
        return 1/(2*m) * np.sum(np.square(self.predict(x) - y))

    def fit(self, x, y):
        m, n = x.shape
        self.W = np.random.randn(n)
        self.b = np.random.randn(1)

        for i in range(self.n_iters):
            self.cost_history[i] = self.cost(x, y, m)
            y_pred = sigmoid(x.dot(self.W) + self.b)
            self.W -= self.learning_rate * ((1 / m) * np.dot(x.T, (y_pred - y)))
            self.b -= self.learning_rate * ((1 / m) * np.sum(y_pred - y))

    def predict(self, x):
        y_pred = sigmoid(x.dot(self.W) + self.b)
        return np.array([1 if i > .5 else 0 for i in y_pred]) 