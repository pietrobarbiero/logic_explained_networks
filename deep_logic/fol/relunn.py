from typing import Tuple, List
import collections

import torch
import numpy as np
from sympy import simplify_logic

from ..utils.base import collect_parameters
from ..utils.relunn import get_reduced_model


def generate_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                                sample_id: int, concept_names: List = None,
                                device: torch.device = torch.device('cpu')) -> str:
    """
    Generate the FOL formula for a specific input.

    :param model: pytorch model
    :param x: all input samples in the dataset
    :param y: all labels in the dataset
    :param sample_id: number of the element to analyse
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: local explanation
    """
    weights, _ = collect_parameters(model, device)
    weights = weights[0][0]

    # if all the weights are zero, then the explanation is always 'False'
    w_abs = np.abs(weights)
    w_max = np.max(w_abs)
    if w_max == 0:
        return 'False'

    # sort features by weight importance
    sorted_features = np.argsort(-w_abs)

    class_id = y[sample_id]  # TODO: NOT WORKING IF Y MULTILABEL = [[0, 1], [1, 0], ...]
    class_mask = (y!=class_id).squeeze()
    x_opposite = x[class_mask]
    y_opposite = y[class_mask]
    x_sample = x[sample_id].unsqueeze(0)
    y_sample = y[sample_id].unsqueeze(0)

    x_data_sample = torch.cat([x_opposite, x_sample])
    y_data_sample = torch.cat([y_opposite, y_sample])

    # build explanation
    explanation = ''
    for fj in sorted_features:
        xij = x_sample[:, fj]

        if explanation:
            explanation += ' & '

        term = f'feature{fj:05}'
        if xij >= 0.5:
            explanation += term
        else:
            explanation += f'~{term}'

        accuracy, _ = test_explanation(explanation, x_data_sample, y_data_sample)
        if accuracy == 1:
            break

    if concept_names:
        explanation = replace_names(explanation, concept_names)

    return explanation


def test_explanation(explanation: str, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, np.ndarray]:
    minterms = str(explanation).split(' | ')
    x_bool = x.detach().numpy() > 0.5
    predictions = np.zeros(x.shape[0], dtype=bool)
    for minterm in minterms:
        minterm = minterm.replace('(', '').replace(')', '').split(' & ')
        local_predictions = np.ones(x.shape[0], dtype=bool)
        for terms in minterm:
            terms = terms.split('feature')
            if terms[0] == '~':
                local_predictions *= ~x_bool[:, int(terms[1])]
            else:
                local_predictions *= x_bool[:, int(terms[1])]

        predictions += local_predictions

    accuracy = sum(predictions == y.detach().numpy().squeeze()) / len(predictions)
    return accuracy, predictions


def replace_names(explanation: str, concept_names: List[str]) -> str:
    feature_abbreviations = [f'feature{i:05}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_abbr, f_name))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation


# TODO: update function with new signature of local explanation
def combine_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                               topk_explanations: int = 2, concept_names: List = None,
                               device: torch.device = torch.device('cpu')) -> Tuple[str, np.array, collections.Counter]:
    """
    Generate a global explanation combining local explanations.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param topk_explanations: number of most common local explanations to combine in a global explanation (it controls the complexity of the global explanation)
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: Global explanation, predictions, and ranking of local explanations
    """
    local_explanations = []
    local_explanations_translated = []
    # TODO: multi class
    for sample_id, (xi, yi) in enumerate(zip(x, y)):
        # get reduced model for each sample
        model_reduced = get_reduced_model(model, xi)
        output = model_reduced(xi)

        # generate local explanation only if the prediction is correct
        if output > 0.5 and (output > 0.5) == yi:
            local_explanation = generate_local_explanations(model_reduced, x, y, sample_id,
                                                            concept_names=None, device=device)
            local_explanations.append(local_explanation)
            local_explanation_translated = generate_local_explanations(model_reduced, x, y, sample_id,
                                                                       concept_names=concept_names, device=device)
            local_explanations_translated.append(local_explanation_translated)

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

    # predictions based on FOL formula
    accuracy, predictions = test_explanation(global_explanation_simplified, x, y)

    # replace concept names
    if concept_names:
        global_explanation_simplified_str = replace_names(global_explanation_simplified_str, concept_names)

    return global_explanation_simplified_str, predictions, counter_translated
