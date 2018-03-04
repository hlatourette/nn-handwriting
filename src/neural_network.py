import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.i_nodes = inputNodes
        self.h_nodes = hiddenNodes
        self.o_nodes = outputNodes
        self.alpha = learningRate
        self.w_ih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_ho = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

    def train(self, inputs, target):
        inputs = np.array(inputs, ndmin=2).T
        target = np.array(target, ndmin=2).T
        hidden_output = hidden_output = self._activation(np.dot(self.w_ih, inputs))
        output = self._activation(np.dot(self.w_ho, hidden_output))
        output_err = (target - output)
        hidden_err = np.dot(self.w_ho.T, output_err)
        self.w_ho += self.alpha * np.dot(output_err * output * (1.0 - output), hidden_output.T)
        self.w_ih += self.alpha * np.dot(hidden_err * hidden_output * (1.0 - hidden_output), inputs.T)

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        hidden_output = self._activation(np.dot(self.w_ih, inputs))
        return self._activation(np.dot(self.w_ho, hidden_output))

    def _activation(self, X):
        return scipy.special.expit(X)