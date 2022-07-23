import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def initializeRandomWeights(self, rng):
        for layer in self.layers:
            layer.initializeRandomWeights(rng)

    def loadWeights(self):
        pass

    def saveWeights(self):
        pass

    def inference(self, input, argmax=False):
        outputs = None

        _i = lambda x : min(input.shape[0], x + 6000)
        for i in range(0, input.shape[0], 6000):
            output = input[i:_i(i)]
            for layer in self.layers:
                output = layer.inference(output)

            if i==0:
                outputs = output
            else:
                outputs = np.concatenate( (outputs, output) )

        if argmax:
            return np.argmax(outputs, axis=1).reshape(input.shape[0], 1)
        return outputs

    def forwardProp(self, input):
        output = input
        for layer in self.layers:
            output = layer.forwardProp(output)
        return output

    def computeGradients(self, error):
        tempError = error
        gradients = []
        for layer in reversed(self.layers):
            layerGradient, tempError = layer.backProp(tempError)
            gradients.append(layerGradient)
        return list(reversed(gradients))

    def regularizerCost(self):
        return np.sum([layer.regularizerCost() for layer in self.layers])