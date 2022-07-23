import numpy as np

class l2cost:
    @staticmethod
    def f(X, Y):
        return np.linalg.norm(X-Y, ord="fro")/X.shape[0]

    @staticmethod
    def backward(X, Y):
        return 2*(X-Y)

class NLLHcost:
    @staticmethod
    def f(X, Y):
        return -np.sum(np.multiply(Y, np.log(X + 1e-9)))/X.shape[0]

    @staticmethod
    def backward(X, Y):
        return -Y/(X + 1e-9)

class NLLHcost2:
    @staticmethod
    def f(X, Y):
        return -np.sum(np.multiply(Y, np.log(X + 1e-9)))/X.shape[0]

    @staticmethod
    def backward(X, Y):
        return Y