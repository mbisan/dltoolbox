import numpy as np

from .layer import Layer

import tensorflow as tf

class conv2DLayer(Layer):
    def __init__(self, inputs, filters, filter_size, regularizer=None):
        super().__init__()
        self._filters = np.empty((filter_size[0], filter_size[1], inputs, filters), dtype=np.float32)
        self._regularizer = regularizer

    def forwardProp(self, layerInput):
        # layerInput: samples x (width x height) x feature_maps
        # output: samples x (width' x height') x filters
        self._lastInput = layerInput
        return tf.nn.conv2d(tf.constant(self._lastInput),
                            tf.constant(self._filters),
                            strides=(1,1),
                            data_format="NHWC",
                            padding="VALID").numpy()

    def initializeRandomWeights(self, rng):
        newFilters = np.array([rng() for i in range(self._filters.size)], dtype=np.float32).reshape(self._filters.shape)
        self.updateWeights(newFilters)

    def updateWeights(self, newFilters):
        self._filters = newFilters
        return True

    def weights(self):
        return self._filters

    def regularizerCost(self):
        return self._regularizer.f(self._filters) if self._regularizer is not None else 0

    def backProp(self, error):
        # error: samples x (width' x height') x filters_o

        input_tensor = tf.constant(self._lastInput)
        error_tensor = tf.constant(error)
        filters_tensor = tf.constant(self._filters)

        # filters_i x (width x height) x samples
        reshaped_input = tf.transpose(input_tensor, perm=(3,1,2,0))
        # (width' x height') x samples x filters_o
        error_filters = tf.transpose(error_tensor, perm=(1,2,0,3))
        # (filter_h x filter_w) x filters_o x filters_i
        flipped_filters = tf.transpose(tf.reverse(filters_tensor, axis=(0,1)), perm=(0,1,3,2))

        pad = [ [0, 0],
                [self._filters.shape[0] - 1, self._filters.shape[0] - 1],
                [self._filters.shape[1] - 1, self._filters.shape[1] - 1],
                [0, 0]]

        # filters_i x (filter_h x filter_w) x filters_o
        filter_gradients = tf.nn.conv2d(reshaped_input,
                                        error_filters,
                                        strides=1,
                                        data_format="NHWC",
                                        padding="VALID")

        # (filter_h x filter_w) x filters_i x filters_o
        filter_gradients = tf.transpose(filter_gradients, perm=(1,2,0,3)).numpy() # /error.shape[0]

        # samples x (width x height) x filter_i
        error_gradients = tf.nn.conv2d(error_tensor,
                                       flipped_filters,
                                       strides=1,
                                       data_format="NHWC",
                                       padding=pad).numpy()

        if self._regularizer is not None:
            filter_gradients += self._regularizer.df(self._filters)

        return filter_gradients, error_gradients
