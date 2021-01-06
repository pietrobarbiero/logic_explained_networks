from typing import List

import numpy as np
from itertools import product


def _forward(X: np.array, weights: np.array, bias: np.array) -> np.array:
    """
    Simulate the forward pass on one layer.

    :param X: input matrix.
    :param weights: weight matrix.
    :param bias: bias vector.
    :return: layer output
    """
    a = np.matmul(weights, np.transpose(X))
    b = np.reshape(np.repeat(bias, np.shape(X)[0], axis=0), np.shape(a))
    output = sigmoid_activation(a + b)
    y_pred = np.where(output < 0.5, 0, 1)
    return y_pred



def get_nonpruned_weights(weight_matrix: np.array, fan_in: int) -> np.array:
    """
    Get non-pruned weights.

    :param weight_matrix: weight matrix of the reasoning network; shape: $h_{i+1} \times h_{i}$.
    :param fan_in: number of incoming active weights for each neuron in the network.
    :return: non-pruned weights
    """
    n_neurons = weight_matrix.shape[0]
    weights_active = np.zeros((n_neurons, fan_in))
    for i in range(n_neurons):
        nonpruned_positions = np.nonzero(weight_matrix[i])
        weights_active[i] = (weight_matrix)[i, nonpruned_positions]
    return weights_active


def build_truth_table(fan_in: int) -> np.array:
    """
    Build the truth table taking into account non-pruned features only,

    :param fan_in: number of incoming active weights for each neuron in the network.
    :return: truth table
    """
    items = []
    for i in range(fan_in):
        items.append([0, 1])
    truth_table = list(product(*items))
    return np.array(truth_table)


def get_nonpruned_positions(weights: List[np.array], neuron_list: List[int]) -> List:
    """
    Get the list of the position of non-pruned weights.

    :param weights: list of the weight matrices of the reasoning network; shape: $h_{i+1} \times h_{i}$.
    :param neuron_list: list containing the number of neurons for each layer of the network.
    :return: list of the position of non-pruned weights
    """
    nonpruned_positions = []
    for j in range(len(weights)):
        non_pruned_position_layer_j = []
        for i in range(neuron_list[j]):
            non_pruned_position_layer_j.append(np.nonzero(weights[j][i]))
        nonpruned_positions.append(non_pruned_position_layer_j)

    return nonpruned_positions


def count_neurons(weights: List[np.array]) -> List[int]:
    """
    Count the number of neurons for each layer of the neural network.

    :param weights: list of the weight matrices of the reasoning network; shape: $h_{i+1} \times h_{i}$.
    :return: number of neurons for each layer of the neural network
    """
    n_layers = len(weights)
    neuron_list = np.zeros(n_layers, dtype=int)
    for j in range(n_layers):
        # for each layer of weights,
        # get the shape of the weight matrix (number of output neurons)
        neuron_list[j] = np.shape(weights[j])[0]
    return neuron_list


def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))
