import copy
from typing import Tuple, List

import torch
import numpy as np


def test_explanation(explanation: str, target_class: int, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, np.ndarray]:
    """
    Test explanation

    :param explanation: formula
    :param target_class: class ID
    :param x: input data
    :param y: input labels
    :return: Accuracy of the explanation and predictions
    """
    minterms = str(explanation).split(' | ')
    x_bool = x.cpu().detach().numpy() > 0.5
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

    if len(y.squeeze().shape) > 1:
        y = torch.argmax(y, dim=1) == target_class

    accuracy = sum(predictions == y.cpu().detach().numpy().squeeze()) / len(predictions)
    return accuracy, predictions


def replace_names(explanation: str, concept_names: List[str]) -> str:
    """
    Replace names of concepts in a formula.

    :param explanation: formula
    :param concept_names: new concept names
    :return: Formula with renamed concepts
    """
    feature_abbreviations = [f'feature{i:010}' for i in range(len(concept_names))]
    mapping = []
    for f_abbr, f_name in zip(feature_abbreviations, concept_names):
        mapping.append((f_abbr, f_name))

    for k, v in mapping:
        explanation = explanation.replace(k, v)

    return explanation


def simplify_formula(explanation: str, model: torch.nn.Module,
                     x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor,
                     target_class: int) -> str:
    """
    Simplify formula to a simpler one that is still coherent.

    :param explanation: local formula to be simplified.
    :param model: torch model.
    :param x: input data.
    :param y: target labels.
    :param x_sample: sample associated to the local formula.
    :param target_class: target class
    :return: Simplified formula
    """
    y_pred_sample = (model((x_sample > 0.5).to(torch.float)) > 0.5).to(torch.float)
    if y_pred_sample != target_class:
        return ''

    mask = (y != y_pred_sample).squeeze()
    x_validation = torch.cat([x[mask], x_sample]).to(torch.bool)
    y_validation = torch.cat([y[mask], y_pred_sample]).squeeze()
    for term in explanation.split(' & '):
        explanation_simplified = copy.deepcopy(explanation)

        if explanation_simplified.endswith(f'{term}'):
            explanation_simplified = explanation_simplified.replace(f' & {term}', '')
        else:
            explanation_simplified = explanation_simplified.replace(f'{term} & ', '')

        if explanation_simplified:
            accuracy, _ = test_explanation(explanation_simplified, target_class, x_validation, y_validation)
            if accuracy == 1:
                explanation = copy.deepcopy(explanation_simplified)

    return explanation
