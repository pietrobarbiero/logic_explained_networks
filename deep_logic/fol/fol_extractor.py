from typing import List

import torch
import numpy as np
from sympy.logic import simplify_logic

from ._utils import count_neurons, get_nonpruned_positions, \
    build_truth_table, get_nonpruned_weights, _forward
from ..utils import collect_parameters


def generate_fol_explanations(model: torch.nn.Module, device: torch.device = torch.device('cpu')) -> List[str]:
    """
    Generate the FOL formulas corresponding to the parameters of a reasoning network.

    :param model: pytorch model
    :param device: cpu or cuda device
    :return: first-order logic formulas
    """
    weights, bias = collect_parameters(model, device)
    assert len(weights) == len(bias)

    # count number of layers of the reasoning network
    n_layers = len(weights)
    fan_in = np.count_nonzero((weights[0])[0, :])
    n_features = np.shape(weights[0])[1]

    # create fancy feature names
    feature_names = list()
    for k in range(n_features):
        feature_names.append("f" + str(k + 1))

    # count the number of hidden neurons for each layer
    neuron_list = count_neurons(weights)
    # get the position of non-pruned weights
    nonpruned_positions = get_nonpruned_positions(weights, neuron_list)

    # generate the query dataset, i.e. a truth table
    truth_table = build_truth_table(fan_in)

    # simulate a forward pass using non-pruned weights only
    predictions = list()
    for j in range(n_layers):
        weights_active = get_nonpruned_weights(weights[j], fan_in)
        y_pred = _forward(truth_table, weights_active, bias[j])
        predictions.append(y_pred)

    formulas = None
    for j in range(n_layers):
        formulas = list()
        for i in range(neuron_list[j]):
            formula = compute_fol_formula(truth_table, predictions[j][i], feature_names, nonpruned_positions[j][i][0])
            formulas.append(f'({formula})')

        # the new feature names are the formulas we just computed
        feature_names = formulas
    return formulas


def compute_fol_formula(truth_table: np.array, predictions: np.array, feature_names: List[str],
                        nonpruned_positions: List[np.array]) -> str:
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
    formula = formula.replace('~(True)', 'False')
    formula = formula.replace('~(False)', 'True')

    # simplify formula
    simplified_formula = simplify_logic(formula)
    return str(simplified_formula)
