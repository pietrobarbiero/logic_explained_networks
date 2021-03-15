import copy
from typing import Tuple, List

import torch
import numpy as np
from sklearn.metrics import accuracy_score

from ..utils.base import to_categorical
from ..utils.metrics import Metric, Accuracy


def test_multi_class_explanation(explanation: str, target_class: int, x: torch.Tensor, y: torch.Tensor,
                                 give_local: bool = False, metric: Metric = Accuracy(), concept_names: list = None) \
        -> Tuple[float, np.ndarray]:
    y_single_class = y.eq(target_class)
    return test_explanation(explanation, target_class=1, x=x, y=y_single_class,
                            give_local=give_local, metric=metric, concept_names=concept_names)


def test_explanation(explanation: str, target_class: int, x: torch.Tensor, y: torch.Tensor,
                     give_local: bool = False, metric: Metric = Accuracy(), concept_names: list = None) \
        -> Tuple[float, np.ndarray]:
    """
    Test explanation

    :param explanation: formula
    :param target_class: class ID
    :param x: input data
    :param y: input labels (categorical, NOT one-hot encoded)
    :param give_local: if true will return local predictions
    :return: Accuracy of the explanation and predictions
    """

    assert concept_names is not None or "feature" in explanation or explanation == "", \
        "Concept names must be given when present in the formula"
    if concept_names is not None:
        for i, concept_name in enumerate(concept_names):
            explanation = explanation.replace(concept_name, f"feature{i:010}")

    if explanation == '':
        local_predictions = [torch.tensor(np.empty_like(y))]
        predictions = torch.cat(local_predictions).eq(target_class).cpu().detach().numpy()
        accuracy = 0.0
        return accuracy, torch.stack(local_predictions, dim=0).sum(dim=0) > 0 if give_local else predictions
    if explanation == "(True)" or explanation == "True":
        local_predictions = [torch.tensor(np.ones_like(y))]
        predictions = torch.cat(local_predictions).eq(target_class).cpu().detach().numpy()
    elif explanation == "(False)" or explanation == "False":
        local_predictions = [torch.tensor(np.zeros_like(y))]
        predictions = torch.cat(local_predictions).eq(target_class).cpu().detach().numpy()
    else:
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
        predictions = (torch.stack(local_predictions, dim=0).sum(dim=0) > 0).cpu().detach().numpy()

    y = to_categorical(y).eq(target_class).cpu().detach().numpy()

    accuracy = metric(y, predictions)
    return accuracy, torch.stack(local_predictions, dim=0).sum(dim=0) > 0 if give_local else predictions


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


def simplify_formula(explanation: str, x: torch.Tensor, y: torch.Tensor, target_class: int) -> str:
    """
    Simplify formula to a simpler one that is still coherent.

    :param explanation: local formula to be simplified.
    :param x: input data.
    :param y: target labels (1D, categorical NOT one-hot encoded).
    :param target_class: target class
    :return: Simplified formula
    """

    base_accuracy, _ = test_explanation(explanation, target_class, x, y, metric=accuracy_score)
    for term in explanation.split(' & '):
        explanation_simplified = copy.deepcopy(explanation)

        if explanation_simplified.endswith(f'{term}'):
            explanation_simplified = explanation_simplified.replace(f' & {term}', '')
        else:
            explanation_simplified = explanation_simplified.replace(f'{term} & ', '')

        if explanation_simplified:
            accuracy, preds = test_explanation(explanation_simplified, target_class, x, y, metric=accuracy_score)
            if accuracy == base_accuracy:
                explanation = copy.deepcopy(explanation_simplified)

    return explanation
