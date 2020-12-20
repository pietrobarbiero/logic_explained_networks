from typing import List

import numpy as np
from sympy import to_cnf
from sympy.logic import simplify_logic

from ._utils import count_neurons, get_nonpruned_positions, \
    build_truth_table, get_nonpruned_weights, forward


def generate_fol_explanations(weights: List[np.array], bias: List[np.array]):
    """
    Generate the FOL formulas corresponding to the parameters of a reasoning network.

    :param weights: list of the weight matrices of the reasoning network; shape: $h_{i+1} \times h_{i}$.
    :param bias: list of the bias vectors of the reasoning network; shape: $h_{i} \times 1$.
    :return:
    """
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
        y_pred = forward(truth_table, weights_active, bias[j])
        predictions.append(y_pred)

    for j in range(n_layers):
        formulas = list()
        for i in range(neuron_list[j]):
            formula = _compute_fol_formula(truth_table, predictions[j][i], feature_names, nonpruned_positions[j][i][0])
            formulas.append(f'({formula})')

        # the new feature names are the formulas we just computed
        feature_names = formulas
    return formulas


def _compute_fol_formula(truth_table, predictions, feature_names, nonpruned_positions):
    """
    Compute First Order Logic formulas.

    :param truth_table: input truth table.
    :param predictions: output predictions for the current neuron.
    :param feature_names: name of the input features.
    :param nonpruned_positions: position of non-pruned weights
    :return:
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


if __name__ == '__main__':
    w1 = np.array([[1, 0, 2, 0, 0], [1, 0, 3, 0, 0], [0, 1, 0, -1, 0]])
    w2 = np.array([[1, 0, -2]])
    b1 = [1, 0, -1]
    b2 = [1]

    w = [w1, w2]
    b = [b1, b2]

    f = generate_fol_explanations(w, b)
    print("Formula: ", f)
