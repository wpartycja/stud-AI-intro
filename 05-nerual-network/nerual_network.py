from itertools import accumulate
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
from sklearn.model_selection import train_test_split
from random import choice
import matplotlib.pyplot as plt


N_INPUT_NEURONS = 11
N_OUTPUT_NEURONS = 6
N_HIDDEN_NEURONS = 100
lr = 0.0003  # learning rate
N_EPOCHS = 200
PROP = 0.75  # size of testing set


class NeuralNetwork():
    def __init__(self, n_inputs, n_hidden_neurons, n_output_neurons, lr):
        """
        n_inputs - number of input nodes
        n_hidden_neurons - number of hidden neurons
        n_output_nodes - number of output neurons
        lr - learning rate (used in back propagation and gradient)
        """
        # initializing
        np.random.seed()
        self.n_inputs = n_inputs
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.lr = lr

        # weights/biases
        # we need two types:
        # *input - hidden* and then *hidden - output*
        self.weights_i_h = 0.1 * np.random.randn(self.n_inputs, self.n_hidden_neurons)
        self.weights_h_o = 0.1 * np.random.randn(self.n_hidden_neurons, self.n_output_neurons)
        self.bias_i_h = 0.1 * np.random.randn(1, self.n_hidden_neurons)
        self.bias_h_o = 0.1 * np.random.randn(1, self.n_output_neurons)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def feed_forward(self, X):
        # operations on hidden layer
        hidden_inputs = np.dot(X, self.weights_i_h) + self.bias_i_h
        hidden_outputs = self.sigmoid(hidden_inputs)

        # as we have results of hidden layer we calculate the output layer
        prediction_inputs = np.dot(hidden_outputs, self.weights_h_o) + self.bias_h_o
        prediction_outputs = self.sigmoid(prediction_inputs)

        return hidden_outputs, prediction_outputs

    def back_propagation(self, X, Y, prediciton_outputs, hidden_outputs):

        # output layer
        derivative1 = Y - prediciton_outputs
        derivative2 = self.d_sigmoid(prediciton_outputs)
        prediciton_error = derivative1 * derivative2

        # hidden layer
        derivative3 = self.d_sigmoid(hidden_outputs)
        hidden_error = np.dot(prediciton_error, self.weights_h_o.T) * derivative3

        # gradients
        prediction_gradient = hidden_outputs.T * prediciton_error
        hidden_gradient = X.T * hidden_error

        return prediciton_error, hidden_error, prediction_gradient, hidden_gradient

    def train(self, features, targets):

        # holders of delta weights
        delta_weights_i_h = np.zeros(self.weights_i_h.shape)
        delta_weights_h_o = np.zeros(self.weights_h_o.shape)
        delta_bias_i_h = np.zeros(self.bias_i_h.shape)
        delta_bias_h_o = np.zeros(self.bias_h_o.shape)

        # calculating for every neuron
        for x, y in zip(features, targets):
            # feed forward
            x = x.reshape(1, x.shape[0])
            hidden_outputs, prediciton_outputs = self.feed_forward(x)

            # back propagation
            prediciton_error, hidden_error, prediction_gradient, hidden_gradient = self.back_propagation(x, y, prediciton_outputs, hidden_outputs)

            # update weights
            delta_weights_i_h += hidden_gradient
            delta_weights_h_o += prediction_gradient
            # bias
            delta_bias_i_h += hidden_error
            delta_bias_h_o += prediciton_error

        self.weights_i_h += self.lr * delta_weights_i_h
        self.weights_h_o += self.lr * delta_weights_h_o
        self.bias_i_h += self.lr * delta_bias_i_h
        self.bias_h_o += self.lr * delta_bias_h_o

    def predict(self, X):
        # returns only the predicitons
        return self.feed_forward(X)[1]

    def test(self, features, targets):
        predictions = self.predict(features)

        n_correct = 0
        n = 0
        for prediction in predictions:
            prediction_value = np.argmax(prediction)+3
            correct = targets[n]

            if prediction_value == correct:
                n_correct += 1
            n += 1

        return n_correct / len(targets)


def split_set(df, ratio):
    train_df, test_df = train_test_split(df, train_size=ratio)
    return train_df, test_df


def load_csv(filename, ratio):
    df = pd.read_csv(filename, sep=";")

    train_df, test_df = split_set(df, ratio)
    train_set = train_df.drop(columns=['quality'], inplace=False).to_numpy()
    qualities = train_df['quality'].to_numpy()
    q_len = qualities.__len__()
    qualities_array = np.zeros((q_len, N_OUTPUT_NEURONS), dtype=int)

    counter = 0
    for row in qualities_array:
        quality = qualities[counter]
        row[quality-3] = 1 # - 3 beacuse of the smjallest class is 3
        counter += 1

    test_set = test_df.drop(columns=['quality'], inplace=False).to_numpy()
    test_outputs = test_df['quality'].to_numpy()
    # x_train, y_train, x_test, y_test
    return train_set, qualities_array, test_set, test_outputs
    
def MSE(x, y):
    return 0.5 * np.mean((x-y)**2)


if __name__ == "__main__":
    network = NeuralNetwork(N_INPUT_NEURONS, N_HIDDEN_NEURONS, N_OUTPUT_NEURONS, lr)
    x_train, y_train, test_set, test_outputs_line = load_csv('winequality-red.csv', PROP)

    # data separation 
    counter = 0
    accuracy_train = []
    accuracy_test = []
    loss_train = []
    loss_test = []

    # simple array with y from training set [5, 6, ...]
    y_train_line = []
    for row in y_train:
        item = np.argmax(row)+3
        y_train_line.append(item)

    # array with types needed to MSE func (as neurons martix)
    test_outputs = np.zeros((test_outputs_line.__len__(), N_OUTPUT_NEURONS), dtype=int)
    i = 0
    for output in test_outputs:
        output[test_outputs_line[i]-3] = 1
        i += 1

    # all magic happens here
    for epoch in range(N_EPOCHS):
        network.train(x_train, y_train)
        if (epoch % (N_EPOCHS//10) == 0):
            print(counter)
            counter += 10

        # updating to make plots
        accuracy_train.append(network.test(x_train, y_train_line))
        accuracy_test.append(network.test(test_set, test_outputs_line))
        loss_train.append(MSE(network.predict(x_train), y_train))
        loss_test.append(MSE(network.predict(test_set), test_outputs))

    # plots creating
    plt.figure(0)
    plt.plot(accuracy_train)
    plt.plot(accuracy_test)

    plt.figure(1)
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.show()
