from turtle import forward
import numpy as np

class Layer:
    def __init__(self):
        pass

    def initializeRandomWeights(self, rng=None):
        pass

    def updateWeights(self, newWeights=None):
        pass

    def forwardProp(self, layerInput):
        pass

    def inference(self, layerInput):
        return self.forwardProp(layerInput)

    def weights(self):
        return np.array([])

    def regularizerCost(self):
        return 0