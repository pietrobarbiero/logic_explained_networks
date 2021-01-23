from typing import Tuple, List
import collections

import torch
import numpy as np
from sympy import simplify_logic

from .base import replace_names, test_explanation, simplify_formula
from .sigmoidnn import _build_truth_table
from ..utils.base import collect_parameters


def combine_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                               target_class: int, topk_explanations: int = 2, concept_names: List = None,
                               device: torch.device = torch.device('cpu')) -> Tuple[str, np.array, collections.Counter]:
    """
    Generate a global explanation combining local explanations.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param target_class: class ID
    :param topk_explanations: number of most common local explanations to combine in a global explanation (it controls the complexity of the global explanation)
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: Global explanation, predictions, and ranking of local explanations
    """
    local_explanations = []
    local_explanations_translated = []
    for sample_id, (xi, yi) in enumerate(zip(x, y)):
        # get prediction for each sample
        output = model(xi)
        pred_class = torch.argmax(output)
        true_class = torch.argmax(yi)

        # generate local explanation only if the prediction is correct
        if pred_class.eq(true_class).item() and pred_class.eq(target_class).item():
            local_explanation = explain_local(model, x, y, xi, target_class, concept_names=None, device=device)
            local_explanations.append(local_explanation)
            local_explanation_translated = replace_names(local_explanation, concept_names)
            local_explanations_translated.append(local_explanation_translated)

    if len(local_explanations) == 0:
        return '', np.array, collections.Counter()

    # get most frequent local explanations
    counter = collections.Counter(local_explanations)
    if len(counter) < topk_explanations:
        topk_explanations = len(counter)
    most_common_explanations = []
    for explanation, _ in counter.most_common(topk_explanations):
        most_common_explanations.append(explanation)

    counter_translated = collections.Counter(local_explanations_translated)

    # the global explanation is the disjunction of local explanations
    global_explanation = ' | '.join(most_common_explanations)
    global_explanation_simplified = simplify_logic(global_explanation, 'dnf', force=True)
    global_explanation_simplified_str = str(global_explanation_simplified)

    if not global_explanation_simplified_str:
        return '', np.array, collections.Counter()

    # predictions based on FOL formula
    accuracy, predictions = test_explanation(global_explanation_simplified, target_class, x, y)

    # replace concept names
    if concept_names:
        global_explanation_simplified_str = replace_names(global_explanation_simplified_str, concept_names)

    return global_explanation_simplified_str, predictions, counter_translated


def explain_local(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor,
                  target_class: int, concept_names: List = None, device: torch.device = torch.device('cpu')) -> str:
    """
    Generate a local explanation for a single sample.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param x_sample: input for which the explanation is required
    :param target_class: class ID
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: Local explanation
    """
    # get the model prediction on the individual sample
    x_sample = x_sample.unsqueeze(0)
    y_pred_sample = model(x_sample)
    pred_class = torch.argmax(y_pred_sample, dim=1).item()
    if len(y.shape) == 1:
        n_classes = len(torch.unique(y))
    else:
        n_classes = y.shape[1]

    # identify non-pruned features
    w, b = collect_parameters(model, device)
    feature_weights = w[0]
    block_size = feature_weights.shape[0] // n_classes
    feature_used_bool = np.sum(np.abs(feature_weights[pred_class*block_size:(pred_class+1)*block_size]), axis=0) > 0
    feature_used = np.nonzero(feature_used_bool)[0]

    # explanation is the conjunction of non-pruned features
    explanation = ''
    for j in feature_used:
        if explanation:
            explanation += ' & '
        explanation += f'feature{j:010}' if x_sample[:, j] > 0.5 else f'~feature{j:010}'

    simplify = False
    if simplify:
        explanation = simplify_formula(explanation, model, x, y, x_sample, target_class)

    # replace concept placeholders with concept names
    if concept_names:
        explanation = replace_names(explanation, concept_names)

    return explanation


def explain_global(model: torch.nn.Module, n_classes: int,
                   target_class: int, concept_names: List = None,
                   device: torch.device = torch.device('cpu')) -> str:
    """
    Explain the behavior of the model for a whole class.

    :param model: torch model
    :param n_classes: number of classes
    :param target_class: target class
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: Global explanation for a single class
    """
    # identify non pruned features
    w, b = collect_parameters(model, device)
    feature_weights = w[0]
    block_size = feature_weights.shape[0] // n_classes
    feature_used_bool = np.sum(np.abs(feature_weights[target_class*block_size:(target_class+1)*block_size]), axis=0) > 0
    feature_used = np.nonzero(feature_used_bool)[0]

    # if the number of features is too high, then don't even try to get something
    if len(feature_used) > 20:
        return 'The formula is too complex!'

    # build truth table and use it to query the model
    truth_table = _build_truth_table(len(feature_used))
    truth_table_tensor = torch.FloatTensor(truth_table).to(device)
    input_table = torch.zeros((len(truth_table), feature_weights.shape[1])).to(device)
    input_table[:, feature_used] = truth_table_tensor
    predictions = model(input_table)
    if device.type != 'cpu':
        predictions = predictions.cpu()
    predictions = predictions.detach().numpy().squeeze()

    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1) == target_class
    else:
        predictions = predictions > 0.5

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
            feature_name = f'feature{feature_used[j]:010}'

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
    simplified_formula = simplify_logic(formula, 'dnf', force=True)
    simplified_formula = str(simplified_formula)

    # replace concept names
    if concept_names:
        simplified_formula = replace_names(str(simplified_formula), concept_names)

    return simplified_formula
