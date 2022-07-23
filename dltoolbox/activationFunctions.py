import numpy as np

# Definition of the various activation
# functions usable in the models.

class id:
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def df(x):
        return np.ones(x.shape, dtype=np.float32)

    @staticmethod
    def backward(x):
        return np.ones(x.shape, dtype=np.float32)

class sigmoid:
    @staticmethod
    def f(x):
        return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))

    @staticmethod
    def df(x):
        _x = sigmoid.f(x)
        return _x * (1.0 - _x)

    @staticmethod
    def backward(x):
        return x * (1.0 - x)

class relu:
    @staticmethod
    def f(x):
        return x * (x > 0.0)

    @staticmethod
    def df(x):
        return ((x > 0.0) * 1.0).astype(np.float32)

    @staticmethod
    def backward(x):
        return relu.df(x)

# def softplus(x, order=0):
#     if order==0:
#         return np.log(1+np.exp(x))
#     elif order==1:
#         return sigmoid(x)

# def step(x, order=0):
#     if order==0:
#         return relu(x, 1)
#     else:
#         return relu(x, 2)

# def arctan(x, order=0):
#     if order==0:
#         return np.arctan(x)
#     elif order==1:
#         return 1.0/(1.0+np.square(x))

class softMax:
    @staticmethod
    def f(x):
        weights = np.exp(x - x.max(axis=1,keepdims=True), dtype=np.float32)
        return weights/np.sum(weights, axis=1, keepdims=True)

    @staticmethod
    def df(x):
        weights = np.exp(x - x.max(axis=1,keepdims=True), dtype=np.float32)
        output = weights/np.sum(weights, axis=1, keepdims=True)
        return softMax.backward(output)

    @staticmethod
    def backward(x):
        pd = - np.einsum("ij, ik->ijk", x, x)
        i = np.arange(x.shape[1])
        pd[:,i,i] += x
        return pd