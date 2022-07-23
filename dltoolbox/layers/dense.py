import numpy as np

from dltoolbox.layers.layer import Layer

import tensorflow as tf

class denseLayer(Layer):
    def __init__(self, inputSize, outputSize, regularizer = None):
        super().__init__()
        self._W = np.empty((inputSize + 1, outputSize), dtype=np.float32)
        self._regularizer = regularizer

    def inference(self, layerInput):
        self._lastInput = np.hstack([np.ones(layerInput.shape[0], dtype=np.float32).reshape(layerInput.shape[0], 1), layerInput])
        input_tensor = tf.constant(self._lastInput)
        weights_tensor = tf.constant(self._W)
        return tf.linalg.matmul(input_tensor, weights_tensor).numpy()

    def forwardProp(self, layerInput):
        # lastInput = sample x inputSize + 1
        # lastOutput = sample x outputSize
        self._lastInput = np.hstack([np.ones(layerInput.shape[0], dtype=np.float32).reshape(layerInput.shape[0], 1), layerInput])
        return np.matmul(self._lastInput, self._W)

    def initializeRandomWeights(self, rng):
        newWeights = np.array([rng() for i in range(self._W.size)], dtype=np.float32).reshape(self._W.shape)
        self.updateWeights(newWeights)

    def updateWeights(self, newWeights):
        self._W = newWeights

    def weights(self):
        return self._W

    def regularizerCost(self):
        return self._regularizer.f(self._W) if self._regularizer is not None else 0

    def backProp(self, error):
        # error: samples x outputSize
        # self._lastInput: samples x inputSize + 1

        # inputSize + 1 x outputSize
        partialDerivatives = np.matmul(self._lastInput.T, error) # /error.shape[0]
        # samples x inputSize
        errorPreviousLayer = np.matmul(error, self._W[1:,:].T)

        if self._regularizer is not None:
            regularizer_derivatives = self._regularizer.df(self._W)
            # bias terms are not to be regularized
            regularizer_derivatives[0,:] = 0
            partialDerivatives += regularizer_derivatives

        return partialDerivatives, errorPreviousLayer