import numpy as np

from .layer import Layer

import tensorflow as tf

class maxPool2D(Layer):
    def __init__(self, window_size):
        super().__init__()
        self._WSize = window_size
        self._lastInput = None
        self._repeatOutput = None

    def forwardProp(self, layerInput):
        self._lastInput = tf.constant(layerInput)
        output = tf.nn.max_pool2d(self._lastInput,
                                  ksize=self._WSize,
                                  strides=self._WSize,
                                  padding="VALID",
                                  data_format="NHWC").numpy()
        self._repeatOutput = np.repeat(np.repeat(output, self._WSize[0], axis=1), self._WSize[1], axis=2)
        return output

    def backProp(self, error):
        maxPositions = np.equal(self._repeatOutput, self._lastInput)
        repeatError = np.repeat(np.repeat(error, self._WSize[0], axis=1), self._WSize[1], axis=2)
        return np.array([]), np.multiply(maxPositions, repeatError)