import numpy as np

from utils import r2_score

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    np.random.seed(0)

    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=1234)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    pred_train = regressor.predict(x_train)
    pred_test = regressor.predict(x_test)

    print(f"{r2_score(y_train, pred_train)=}")
    print(f"{r2_score(y_test, pred_test)=}")

    plt.scatter(x_train, y_train)
    plt.scatter(x_test, y_test, c='g')
    plt.plot(x_test, pred_test, c='r')
    plt.show()

    plt.plot(regressor.cost_history)
    plt.show()