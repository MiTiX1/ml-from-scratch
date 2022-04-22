import numpy as np

class LinearRegression:
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
            y_pred = self.predict(x)
            self.W -= self.learning_rate * ((1 / m) * np.dot(x.T, (y_pred - y)))
            self.b -= self.learning_rate * ((1 / m) * np.sum(y_pred - y))

    def predict(self, x):
        return x.dot(self.W) + self.b