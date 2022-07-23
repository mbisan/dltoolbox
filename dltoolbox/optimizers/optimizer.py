import numpy as np

class HyperParameters:
    def __init__(self, batch_size=10, lr=0.001, alpha=0.5, beta1=0.9, beta2=0.999, gamma=0.5, epsilon=1e-6):
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.epsilon = epsilon
        self.batchSize = batch_size

class Optimizer:
    def __init__(self, model, costFunction, hyperparameters):
        self._model = model
        self._costFunction = costFunction
        self._hp = hyperparameters