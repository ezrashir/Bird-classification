import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionPredictor:

    def __init__(self, X, y, iterations=1500, alpha=0.01):
        self.X = X
        self.y = y
        self.n_features = np.size(X, 1)
        self.n_samples = np.size(X, 0)
        self.n_iter = iterations
        self.alpha = alpha

    def computeCost(self, X, y, theta):
        if theta.ndim == 1:
            theta = theta.reshape((-1, 1))
        h = X @ theta
        J = (1 / (2 * self.n_samples)) * np.sum(np.power((h - y), 2))
        return J

    def gradientDescent(self, X, y, theta):
        J_history = np.zeros((self.n_iter, 1))
        for i in range(self.n_iter):
            h = (X @ theta)
            dj = (h-y) @ X
            theta[:] = theta[:] - (self.alpha / self.n_samples) * dj[:]
            J_history[i] = self.computeCost(X, y, theta)
        return J_history, theta

    def predict(self, X_test, plot_J=False):
        theta = np.zeros(self.n_features)
        J_hyst, theta = self.gradientDescent(self.X, self.y, theta)

        if plot_J is True:
            plt.plot(J_hyst, np.arange(self.n_iter), color="orange")
            plt.xlabel("Number of iterations")
            plt.ylabel("Gradient values")
            plt.title("Check gradient descent")
            plt.show()

        return X_test @ theta

class LogisticRegressionPredictor:
    def __init__(self, X, y):
        self.n_features = np.size(X, 1)
        self.n_samples = np.size(X, 0)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.X = X
        self.y = y

    def sigmoid(self, z):
        g = 1/(1 + np.exp(-z))
        return g

    def computeCost(self, theta):
        X = self.X
        y = self.y
        m = self.n_samples
        h = self.sigmoid(X @ theta).reshape((-1, 1))
        J = -1/m * np.sum(y*np.log(h)+(1-y)*np.log(1-h))
        grad = 1/m * np.sum(X*(h-y), axis=0)
        return J, grad

    def costOptimization(self, init_theta):
        from scipy.optimize import fmin_bfgs
        return fmin_bfgs(self.simplifyCost, init_theta, maxiter=400)

    def simplifyCost(self, theta):
        return self.computeCost(theta)[0]

    def predict(self, X_test, init_theta):
        theta = self.costOptimization(init_theta)
        z = self.sigmoid(np.sum(X_test * theta, axis=1))
        m = np.size(z)
        pred = np.zeros((m, 1))
        for i in range(m):
            if z[i] >= 0.5:
                pred[i] = 1
        return z, pred

class RegularizedLogisticRegression:
    def __init__(self, X, y):
        self.n_samples = np.size(X, 0)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.X = X
        self.y = y
        self.n_features = np.size(X, 1)

    def sigmoid(self, z):
        g = 1/(1 + np.exp(-z))
        return g

    def computeCostReg(self, theta, lambdaVal):
        h = self.sigmoid(self.X @ theta).reshape((-1, 1))
        J = -1/self.n_samples * np.sum(self.y*np.log(h)+(1-self.y)*np.log(1-h)) + lambdaVal/(2*self.n_samples)*sum(theta[1:]**2)
        grad = np.zeros(np.size(theta))
        grad[0] = 1/self.n_samples * np.dot(self.X[:, 0], (h-self.y))
        for i in np.arange(1, np.size(theta)):
            grad[i] = float(1/self.n_samples*np.dot(self.X[:, i], (h-self.y)) + lambdaVal/self.n_samples * theta[i])
        return J, grad

    def costOptimization(self, init_theta, lambdaVal):
        from scipy.optimize import fmin_bfgs
        return fmin_bfgs(self.simplifyCost, init_theta, args=(lambdaVal, ), maxiter=400)

    def simplifyCost(self, theta, lamb):
        return self.computeCostReg(theta, lamb)[0]

    def predict(self, X_test, init_theta, lambdaVal):
        theta = self.costOptimization(init_theta, lambdaVal)
        z = self.sigmoid(np.sum(X_test * theta, axis=1))
        m = np.size(z)
        pred = np.zeros((m, 1))
        for i in range(m):
            if z[i] >= 0.5:
                pred[i] = 1
        return z, pred

class MulticlassLinearRegressionPredictor:
    def __init__(self, X, y, lambdaVal, labels_num):
        self.n_samples = np.size(X, 0)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.X = X
        self.y = y
        self.n_features = np.size(X, 1)
        self.k = labels_num
        self.lambd = lambdaVal

    def sigmoid(self, z):
        g = 1 / (1 + np.exp(-z))
        return g

    def computeCostReg(self, theta, y=None):
        if y is None:
            y = self.y
        h = self.sigmoid(self.X @ theta)
        reg_theta = np.concatenate(([0], theta[1:]))
        J = 1 / self.n_samples * np.sum(-np.log(h) * y - (1 - y) * np.log(1 - h)) + self.lambd / (2 * self.n_samples) * sum(reg_theta ** 2)
        grad = 1 / self.n_samples * (h - y) @ self.X + self.lambd / self.n_samples * reg_theta
        return J, grad

    def oneVSall(self):
        from scipy.optimize import fmin_bfgs
        all_theta = np.zeros((self.k, self.n_features))
        for k in range(self.k):
            init_theta = np.zeros(self.n_features)
            y_train = np.array([1 if x == k else 0 for x in self.y])
            all_theta[k, :] = fmin_bfgs(self.simplifyCost, init_theta, fprime=self.simplifyGrad, args=(y_train, ), disp=True,  maxiter=400, full_output=True)[0]
        return all_theta

    def simplifyGrad(self, theta, y):
        return self.computeCostReg(theta, y)[1]

    def simplifyCost(self, theta, y):
        return self.computeCostReg(theta, y)[0]

    def predict(self, X_test):
        X_test = np.hstack((np.ones((np.size(X_test, axis=0), 1)), X_test))
        all_theta = self.oneVSall()
        p = np.zeros(np.size(X_test, 0))
        # return the index of the max element:
        mat = self.sigmoid(X_test @ all_theta.T)
        p[:] = mat.argmax(axis=1)
        return p








