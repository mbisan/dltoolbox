import numpy as np

from .layer import Layer
from dltoolbox.activationFunctions import softMax

class activationLayer(Layer):
    def __init__(self, activationFunction):
        super().__init__()
        self._activation = activationFunction
        self._lastOutput = None

    def forwardProp(self, layerInput):
        # output.shape = layerInput.shape
        self._lastOutput = self._activation.f(layerInput)
        return self._lastOutput

    def backProp(self, error):
        # error = samples x inputSize
        partialDerivatives = self._activation.backward(self._lastOutput)
        errorPreviousLayer = np.multiply(error, partialDerivatives)
        return np.array([]), errorPreviousLayer

class softMaxLayer(activationLayer):
    def __init__(self):
        super().__init__(softMax)

    def backProp(self, error):
        # error = samples x inputSize
        pd = softMax.backward(self._lastOutput) # samples x (inputsize x inputsize)
        errorPreviousLayer = np.matmul(np.expand_dims(error, 1), pd) # samples x inputSize
        return np.array([]), np.squeeze(errorPreviousLayer)

class softMaxLayer2(softMaxLayer):
    def __init__(self):
        super().__init__()

    def backProp(self, error):
        # error = samples x inputSize
        return np.array([]), self._lastOutput - error