import numpy as np

from dltoolbox.optimizers.gradientDescent import GDoptimizer

class Adams(GDoptimizer):
    def __init__(self, model, costFunction, hyperparameters):
        super().__init__(model, costFunction, hyperparameters)
        self._step = 0
        self._deltaS = []
        self._deltaW = []
        for layer in model.layers:
            self._deltaW.append(np.zeros_like(layer.weights()))
            self._deltaS.append(np.zeros_like(layer.weights()))

    def _fitMiniBatch(self, mX, mY):
        self._step +=1

        omX = self._model.forwardProp(mX)
        propagatedError = self._costFunction.backward(omX, mY)
        gradients = self._model.computeGradients(propagatedError)

        lr = self._hp.lr / mX.shape[0]
        lr = self._hp.lr * np.sqrt(1.0 - np.power(self._hp.beta2, self._step)) / (1.0 - np.power(self._hp.beta2, self._step))
        for i, layer in enumerate(self._model.layers):
            self._deltaW[i] = self._hp.beta1 * self._deltaW[i] + (1.0 - self._hp.beta1) * gradients[i]
            self._deltaS[i] = self._hp.beta2 * self._deltaS[i] + (1.0 - self._hp.beta2) * np.square(gradients[i])

            layer.updateWeights(layer.weights() - (lr * self._deltaW[i]) / (np.sqrt(self._deltaS[i]) + self._hp.epsilon) )