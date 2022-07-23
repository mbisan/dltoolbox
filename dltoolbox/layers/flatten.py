import numpy as np

from .layer import Layer

class flatten(Layer):
    def __init__(self):
        # self._lastOutput = None
        self._lastShape = None

    def forwardProp(self, layerInput):
        self._lastShape = layerInput.shape
        return layerInput.reshape((self._lastShape[0], np.prod(self._lastShape[1:])))

    def backProp(self, error):
        return np.array([]), error.reshape(self._lastShape)