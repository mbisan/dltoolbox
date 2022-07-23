import numpy as np

from dltoolbox.optimizers.optimizer import Optimizer

from time import time

class GDoptimizer(Optimizer):
    def __init__(self, model, costFunction, hyperparameters):
        super().__init__(model, costFunction, hyperparameters)

    def accuracy(self, predictedLabels, correctLabels):
        correct = np.sum(np.equal(predictedLabels, correctLabels))
        return (correct/predictedLabels.size)

    def fit(self, X, Y, epochs, show_progress=0, validation_data=(None, None), compute_costs=0, compute_test_costs=0):
        print("Start fitting -> ", end="")
        (X_test, Y_test) = validation_data

        # X and Y must be adapted to the cost function (not checked)

        updates_count = 0

        costHistory = []
        costHistory_test = []
        accuracies = []
        accuracies_test = []

        if compute_costs>0:
            correct_labels = Y.argmax(axis=1)
            train_X = self._model.inference(X)
            cost = self._costFunction.f(train_X, Y)  + self._model.regularizerCost()
            costHistory.append(cost)
            accuracies.append(self.accuracy(train_X.argmax(axis=1), correct_labels))
            print("Initial cost:", cost, end=" ")

        if compute_test_costs>0:
            correct_labels_test = Y_test.argmax(axis=1)
            test_X = self._model.inference(X_test)
            cost_test = self._costFunction.f(test_X, Y_test)  + self._model.regularizerCost()
            costHistory_test.append(cost_test)
            accuracies_test.append(self.accuracy(test_X.argmax(axis=1), correct_labels_test))

        print("Hyperparameters, lr=", self._hp.lr, "Batch-size=", self._hp.batchSize)

        _i = lambda x : min(X.shape[0], x + self._hp.batchSize)
        for epoch in range(1, epochs + 1):
            print("Epoch ", epoch, "/", epochs, ":", sep="")
            a=time()

            for i in range(0, X.shape[0], self._hp.batchSize):
                b=time()
                self._fitMiniBatch(X[i:_i(i)], Y[i:_i(i)])
                updates_count += 1

                if compute_costs>0 and updates_count%compute_costs==0:
                    train_X = self._model.inference(X)
                    cost = self._costFunction.f(train_X, Y)  + self._model.regularizerCost()
                    costHistory.append(cost)
                    accuracies.append(self.accuracy(train_X.argmax(axis=1), correct_labels))

                if compute_test_costs>0 and updates_count%compute_test_costs==0:
                    test_X = self._model.inference(X_test)
                    cost_test = self._costFunction.f(test_X, Y_test)  + self._model.regularizerCost()
                    costHistory_test.append(cost_test)
                    accuracies_test.append(self.accuracy(test_X.argmax(axis=1), correct_labels_test))

                print("\rMini-batch ", int(i/self._hp.batchSize) + 1, "/", int(X.shape[0]/self._hp.batchSize), " in %.3f" % (time()-b), sep="", end="")

            if show_progress>0 and epoch%show_progress==0:
                print(" Last cost:", "%.5f" % costHistory[-1], "Accuracy: %.3f" % accuracies[-1], "in %.3f s" % (time()-a))

        return costHistory, costHistory_test, accuracies, accuracies_test

    def _fitMiniBatch(self, mX, mY):
        omX = self._model.forwardProp(mX)
        propagatedError = self._costFunction.backward(omX, mY)
        gradients = self._model.computeGradients(propagatedError)
        for i, layer in enumerate(self._model.layers):
            layer.updateWeights(layer.weights() - self._hp.lr * gradients[i])