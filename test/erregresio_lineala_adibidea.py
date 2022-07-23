import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

np.random.seed(3141592)
N = 50

class neurona:
    def __init__(self):
        self.W = np.array([np.random.randn(1,1), np.random.randn(1,1)]).reshape(2,1) # 2x1

    def run(self, x):
        return np.matmul(self.W.T,x) # 1x2 & 2x50 -> 1x50

    def gradientea_bberrore(self, x, y):
        # 2x50 & 50x2 & 2x1 - 2x50 & 50x1 -> 2x1
        return 2/y.size*(np.matmul(np.matmul(x,x.T),self.W) - np.matmul(x,y.T))

    def bberrore(self, x, y):
        return 1/y.size*np.linalg.norm(np.matmul(self.W.T,x)-y)

    def train(self, x, y, iter, epsilon):
        weight_history = [self.W]
        error_iter = [self.bberrore(x,y)]

        for i in range(iter):
            self.W = self.W - epsilon*self.gradientea_bberrore(x, y)

            error_iter.append(self.bberrore(x, y))
            weight_history.append(self.W)

        return error_iter, weight_history

def plot_created_data(neur, sample, data):
    linspace = np.arange(0.,5.,5/N).reshape(1,N)
    initial_predictions = neur.run(np.vstack([np.ones([1,N]),linspace]))

    plt.scatter(sample[1], data)
    plt.plot(linspace.reshape(N), initial_predictions.reshape(N), "r-")

    plt.show()

def plot_after_train(neur, sample, data, n, epsilon):
    linspace = np.arange(0.,5.,5/N).reshape(1,N)
    #train neuron over 100 epochs
    error_it, weight_his = neur.train(sample, data, n, epsilon)
    final_predictions = neur.run(np.vstack([np.ones([1,N]),linspace]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(sample[1], data)
    ax1.plot(linspace.reshape(N), final_predictions.reshape(N), "r-")

    ax2.plot(np.arange(0,101,1), error_it)
    ax2.set_yscale("log")
    ax2.set(xlabel="Aroa")
    ax2.set(ylabel="Kostu funtzioa")

    print(weight_his[-1], error_it[-1])

    plt.show()

def plot_solution_normal_eq(sample, data):
    linspace = np.arange(0.,5.,5/N).reshape(1,N)
    wtilda = np.matmul(np.matmul(np.linalg.inv(np.matmul(sample,sample.T)),sample),data.T)

    plt.scatter(sample[1], data)
    plt.plot(linspace.reshape(N), linspace.reshape(N)*wtilda[1] + np.ones(N)*wtilda[0], "r-")

    plt.show()

def plot_error_funct(sample,data):
    linspace = np.meshgrid(np.arange(0,3,3/1000), np.arange(0,3,3/1000))

    @np.vectorize
    def cost_funct(w1,w2):
        w = np.array([w1,w2]).reshape(2,1)
        return 1/N*np.linalg.norm(np.matmul(np.transpose(w),sample)-data)**2

    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    ax.plot_surface(linspace[0], linspace[1], cost_funct(linspace[0],linspace[1]))
    ax.set(xlabel="b")
    ax.set(ylabel="w")
    ax.set(zlabel="Kostu funtzioa")

    plt.show()

if __name__ == "__main__":
    np.random.seed(3141592)

    neuron = neurona()

    sample_points = np.random.uniform(0,5,N).reshape(1,N)
    errorea = np.random.randn(N)
    yhat = 1.5*sample_points + 2 + 0.5*errorea # 1x50
    xhat = np.vstack([np.ones([1,N]),sample_points]) # 2x50

    #plot_after_train(neuron, xhat, yhat, 100, 0.05)
    #plot_created_data(neuron, xhat, yhat)
    plot_error_funct(xhat, yhat)
    #plot_solution_normal_eq(xhat, yhat)