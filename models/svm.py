import cvxopt
import numpy as np
from models.base import Model

# TODO: soft margin SVM
# TODO: multiclass SVM

class SupportVectorMachine(Model):
    """
    Binary Support Vector Machine classifier
    """

    def __init__(self, kernel=None, C=1.0):
        super().__init__()
        self.kernel = kernel if kernel else self._linear_kernel
        self.alpha = None
        self.sv = None
        self.sv_y = None
        self.b = None
        self.w = None # only for linear kernel
        # self.C = C

    def _linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def _polynomial_kernel(self, x, y, p=3):
        return (1 + np.dot(x, y)) ** p
    
    def _gaussian_kernel(self, x, y, sigma=5.0):
        return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
    
    def fit(self, X, y):
        
        n = X.shape[0]

        # optimization problem: 
        # min 1/2 * x^T * P * x + q^T * x
        # s.t. Gx <= h
        #      Ax = b

        # objective function: SVM dual problem
        # minimize: 1/2 sum sum alpha_i * alpha_j * y_i * y_j * K(x_i, x_j) - sum alpha_i
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P[i][j] = y[i] * y[j] * self.kernel(X[i], X[j])

        print(P)
        
        q = np.ones(n) * -1

        # alpha_i >= 0
        G = np.diag(np.ones(n) * -1)
        h = np.zeros(n)

        # sum alpha_i * y_i = 0
        A = y.reshape(1, n)
        b = 0.0

        # convert to cvxopt format
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)

        # solve QP problem
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(sol['x'])

        # support vectors have non zero lagrange multipliers
        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # intercept: b = avg(y_i - sum alpha_j * y_j * K(x_i, x_j))
        self.b = 0
        for i in range(len(self.alpha)):
            self.b += self.sv_y[i]
            for j in range(len(self.alpha)):
                self.b -= self.alpha[j] * self.sv_y[j] * self.kernel(self.sv[i], self.sv[j])

        self.b /= len(self.alpha)

        # weight (only for linear kernel)
        if self.kernel == self._linear_kernel:
            self.w = np.zeros(X.shape[1])
            for i in range(len(self.alpha)):
                self.w += self.alpha[i] * self.sv_y[i] * self.sv[i]

    def predict(self, X):
        
        # formula: sign(sum alpha_i * y_i * K(x_i, x) + b)
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = np.sum(self.alpha * self.sv_y * self.kernel(X[i], self.sv)) + self.b
        
        return np.sign(y_pred)