import numpy as np

from dltoolbox.optimizers.gradientDescent import GDoptimizer

class Momentum(GDoptimizer):
    def __init__(self, model, costFunction, hyperparameters):
        super().__init__(model, costFunction, hyperparameters)
        self._deltaW = []
        for layer in model.layers:
            self._deltaW.append(np.zeros_like(layer.weights()), dtype=np.float32)

    def _fitMiniBatch(self, mX, mY):
        omX = self._model.forwardProp(mX) # computes output of mX od the nn
        propagatedError = self._costFunction.backward(omX, mY)
        gradients = self._model.computeGradients(propagatedError)

        for i, layer in enumerate(self._model.layers):
            self._deltaW[i] = self._hp.beta1 * self._deltaW[i] + (1.0 - self._hp.beta1) * gradients[i]

            layer.updateWeights(layer.weights() - self._hp.lr * self._deltaW[i])