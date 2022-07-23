import numpy as np

class L1regularizer:
    def __init__(self, rate):
        self._rate = rate

    def f(self, W):
        return self._rate * np.sum(np.abs(W))

    def df(self, W):
        return self._rate * np.sign(W)

class L2regularizer:
    def __init__(self, rate):
        self._rate = rate

    def f(self, W):
        return self._rate * np.sum(np.square(W))

    def df(self, W):
        return self._rate * 2 * W