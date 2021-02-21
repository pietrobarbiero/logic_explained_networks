from itertools import product
from typing import List

import torch
import numpy as np
from sympy import sympify
from sympy.logic import simplify_logic

from ..utils.base import collect_parameters


def generate_fol_explanations(model: torch.nn.Module, device: torch.device = torch.device('cpu'),
                              concept_names: list = None, simplify=True) -> List[str]:
    """
    Generate the FOL formulas corresponding to the parameters of a reasoning network.

    :param model: pytorch model
    :param device: cpu or cuda device
    :param concept_names: list of names of the input features
    :return: first-order logic formulas
    """
    weights, bias = collect_parameters(model, device)
    assert len(weights) == len(bias)

    # count number of layers of the reasoning network
    n_layers = len(weights)
    fan_in = np.count_nonzero((weights[0])[0, :])
    n_features = np.shape(weights[0])[1]

    # create fancy feature names
    if concept_names is not None:
        assert len(concept_names) == n_features, "Concept names need to be as much as network input nodes"
        feature_names = concept_names
    else:
        feature_names = list()
        for k in range(n_features):
            feature_names.append(f'feature{k:010}')

    # count the number of hidden neurons for each layer
    neuron_list = _count_neurons(weights)
    # get the position of non-pruned weights
    nonpruned_positions = _get_nonpruned_positions(weights, neuron_list)

    # generate the query dataset, i.e. a truth table
    truth_table = _build_truth_table(fan_in)

    # simulate a forward pass using non-pruned weights only
    predictions = list()
    for j in range(n_layers):
        weights_active = _get_nonpruned_weights(weights[j], fan_in)
        y_pred = _forward(truth_table, weights_active, bias[j])
        predictions.append(y_pred)

    formulas = None
    for j in range(n_layers):
        formulas = list()
        for i in range(neuron_list[j]):
            formula = compute_fol_formula(truth_table, predictions[j][i], feature_names,
                                          nonpruned_positions[j][i][0], simplify=simplify)
            formulas.append(f'({formula})')

        # the new feature names are the formulas we just computed
        feature_names = formulas
    return formulas


def compute_fol_formula(truth_table: np.array, predictions: np.array, feature_names: List[str],
                        nonpruned_positions: List[np.array], simplify=True) -> str:
    """
    Compute First Order Logic formulas.

    :param truth_table: input truth table.
    :param predictions: output predictions for the current neuron.
    :param feature_names: name of the input features.
    :param nonpruned_positions: position of non-pruned weights
    :return: first-order logic formula
    """
    # select the rows of the input truth table for which the output is true
    X = truth_table[np.nonzero(predictions)]

    # if the output is never true, then return false
    if np.shape(X)[0] == 0: return "False"

    # if the output is never false, then return true
    if np.shape(X)[0] == np.shape(truth_table)[0]: return "True"

    # compute the formula
    formula = ''
    n_rows, n_features = X.shape
    for i in range(n_rows):
        # if the formula is not empty, start appending an additional term
        if formula != '':
            formula = formula + "|"

        # open the bracket
        formula = formula + "("
        for j in range(n_features):
            # get the name (column index) of the feature
            feature_name = feature_names[nonpruned_positions[j]]

            # if the feature is not active,
            # then the corresponding predicate is false,
            # then we need to negate the feature
            if X[i][j] == 0:
                formula += "~"

            # append the feature name
            formula += feature_name + "&"

        formula = formula[:-1] + ')'

    # replace "not True" with "False" and vice versa
    formula = formula.replace('~(True)', '(False)')
    formula = formula.replace('~(False)', '(True)')

    # simplify formula
    try:
        if eval(formula) == True or eval(formula) == False:
            formula = str(eval(formula))
            assert not formula == "-1" and not formula == "-2", "Error in evaluating formulas"
    except:
        formula = simplify_logic(formula, force=simplify)
    return str(formula)


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
    output = _sigmoid_activation(a + b)
    y_pred = np.where(output < 0.5, 0, 1)
    return y_pred


def _get_nonpruned_weights(weight_matrix: np.array, fan_in: int) -> np.array:
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


def _build_truth_table(fan_in: int) -> np.array:
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


def _get_nonpruned_positions(weights: List[np.array], neuron_list: List[int]) -> List:
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


def _count_neurons(weights: List[np.array]) -> List[int]:
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


def _sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))
