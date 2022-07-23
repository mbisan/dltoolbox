import numpy as np

from .layer import Layer

class dropoutLayer(Layer):
    def __init__(self, rate):
        self._rate = rate
        self._lastActive = None

    def forwardProp(self, layerInput):
        self._lastActive = np.random.binomial(1, 1 - self._rate, layerInput.shape).astype(np.float32)
        return layerInput * self._lastActive

    def inference(self, layerInput):
        # no dropout on inference
        return layerInput

    def backProp(self, error):
        # only gradients through active outputs survive
        return np.array([]), error * self._lastActive