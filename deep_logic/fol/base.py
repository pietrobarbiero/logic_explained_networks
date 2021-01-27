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
    x = x > 0.5
    local_predictions = []
    for minterm in minterms:
        minterm = minterm.replace('(', '').replace(')', '').split(' & ')
        features = []
        for terms in minterm:
            terms = terms.split('feature')
            if terms[0] == '~':
                features.append(~x[:, int(terms[1])])
            else:
                features.append(x[:, int(terms[1])])

        local_prediction = torch.stack(features, dim=0).prod(dim=0)
        local_predictions.append(local_prediction)

    if len(y.squeeze().shape) > 1:
        predictions = (torch.stack(local_predictions, dim=0).sum(dim=0) > 0).cpu().detach().numpy()
        y = torch.argmax(y, dim=1).eq(target_class)
    else:
        predictions = (torch.stack(local_predictions, dim=0).sum(dim=0) > 0).eq(target_class).cpu().detach().numpy()
        y = y > 0.5

    y = y.cpu().detach().numpy().squeeze()

    accuracy = sum(predictions == y) / len(predictions)
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
    if y_pred_sample.numel() > 1:
        y_pred_sample = torch.argmax(y_pred_sample).unsqueeze(0)
    elif len(y_pred_sample.shape) == 1:
        y_pred_sample = y_pred_sample.unsqueeze(0)

    if len(y.squeeze().shape) > 1:
        y = torch.argmax(y, dim=1)

    if not y_pred_sample.eq(target_class):
        return ''

    if len(x_sample.shape) == 1:
        x_sample = x_sample.unsqueeze(0)
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
