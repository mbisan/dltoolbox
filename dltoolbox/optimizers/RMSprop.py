import numpy as np

from dltoolbox.optimizers.gradientDescent import GDoptimizer

class RMSprop(GDoptimizer):
    def __init__(self, model, costFunction, hyperparameters):
        super().__init__(model, costFunction, hyperparameters)
        self._deltaS = []
        for layer in model.layers:
            self._deltaS.append(np.zeros_like(layer.weights()))

    def _fitMiniBatch(self, mX, mY):
        omX = self._model.forwardProp(mX) # computes output of mX od the nn
        propagatedError = self._costFunction.backward(omX, mY)
        gradients = self._model.computeGradients(propagatedError)

        for i, layer in enumerate(self._model.layers):
            self._deltaS[i] = self._hp.beta1 * self._deltaS[i] + (1.0 - self._hp.beta1) * np.square(gradients[i])

            layer.updateWeights(layer.weights() - self._hp.lr * gradients[i] / np.sqrt(self._deltaS[i] + self._hp.epsilon) )