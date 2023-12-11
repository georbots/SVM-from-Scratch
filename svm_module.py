import numpy as np
from cvxopt import matrix, solvers


class SVM:
    def __init__(self, C, kernel='rbf', gamma=1):
        self.coef_ = None
        self.intercept_ = None
        self.alphas = None
        self.support_vectors = None
        self.C = C
        self.sv_y = None

        self.kernel = kernel
        self.gamma = gamma

    def gaussian_kernel(self, x, z, sigma):
        n = x.shape[0]
        m = z.shape[0]
        xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
        zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))
        return np.exp(-self.gamma*(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * sigma ** 2))

    @staticmethod
    def linear_kernel(x, z):
        return np.matmul(x, z.T)

    def prepare_qp_input(self, X, y):
        m = X.shape[0]

        if self.kernel == 'rbf':
            K = self.gaussian_kernel(X, X, sigma=1)
        elif self.kernel == 'linear':
            K = self.linear_kernel(X, X)
        else:
            raise ValueError('Invalid kernel specified.')

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(m))
        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(0.0)
        G = matrix(np.vstack([-np.eye(m), np.eye(m)]))
        h = matrix(np.hstack([np.zeros(m), np.ones(m) * self.C]))
        return P, q, G, h, A, b

    def train(self, X, y):
        P, q, G, h, A, b = self.prepare_qp_input(X, y)
        sol = solvers.qp(P, q, G, h, A, b, kktsolver='ldl')
        alphas = np.array(sol["x"])

        ind = (alphas > 1e-5).flatten()
        self.support_vectors = X[ind]

        if self.kernel == 'linear':
            self.coef_ = np.dot((y * alphas).T, X)[0]
            self.intercept_ = np.mean(y[ind] - np.dot(X[ind], self.coef_))

        elif self.kernel == 'rbf':
            self.sv_y = y[ind]
            self.alphas = alphas[ind]
            self.intercept_ = self.sv_y - np.sum(
                self.gaussian_kernel(self.support_vectors, self.support_vectors, sigma=1) * self.alphas * self.sv_y,
                axis=0)
            self.intercept_ = np.sum(self.intercept_) / self.intercept_.size

    def predict(self, input_data):
        if self.intercept_ is None:
            raise ValueError('The model has not been trained.')

        if self.kernel == 'linear':
            decision_function = np.dot(input_data, self.coef_) + self.intercept_
        elif self.kernel == 'rbf':
            decision_function = np.sum(
                self.gaussian_kernel(self.support_vectors, input_data, sigma=1) * self.alphas * self.sv_y,
                axis=0) + self.intercept_
        else:
            raise ValueError('Not applicable kernel')

        return np.sign(decision_function).astype(int).flatten()

