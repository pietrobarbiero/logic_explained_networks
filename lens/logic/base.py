import copy
from typing import Tuple, List

import torch
import numpy as np
from sympy import to_dnf

from ..utils.base import to_categorical
from ..utils.metrics import Metric, Accuracy


def test_explanation(explanation: str, target_class: int, x: torch.Tensor, y: torch.Tensor,
                     give_local: bool = False, metric: Metric = Accuracy(), concept_names: list = None,
                     inequalities=False) \
        -> Tuple[float, torch.Tensor]:
    """
    Test explanation

    :param explanation: formula
    :param target_class: class ID
    :param x: input data
    :param y: input labels (categorical, NOT one-hot encoded)
    :param give_local: if true will return local predictions
    :param inequalities: if true check when the inequalities are correct x > 0.2 instead of x > 0.5
    :return: Accuracy of the explanation and predictions
    """

    assert concept_names is not None or "feature" in explanation or explanation == "", \
        "Concept names must be given when present in the formula"
    if explanation == '':
        local_predictions = [torch.empty_like(y)]
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
        if concept_names is not None:
            for i, concept_name in enumerate(concept_names):
                explanation = explanation.replace(concept_name, f"feature{i:010}")
            absent_concepts = [term for term in explanation.split(" ") if term != "&" and term != "|" and
                               "feature" not in term and "True" not in term and "False" not in term and "" not in term]
            assert len(
                absent_concepts) == 0, "Some of the concepts in the explanations have not been passed to the function"
        if not inequalities:
            explanation = to_dnf(explanation)
            x = x > 0.5
        min_terms = str(explanation).split(' | ')
        local_predictions = []
        for min_term in min_terms:
            min_term = min_term.replace('(', '').replace(')', '').split(' & ')
            features = []
            for terms in min_term:
                if inequalities:
                    terms = terms.replace("feature", "").split(" ")
                    if len(terms) == 3:
                        feature_num = int(terms[0])
                        sign = terms[1]
                        geq = "=" in sign
                        sign = sign.replace("=", "")
                        threshold = float(terms[2])
                        assert sign == ">" or sign == "<", f"Invalid sign {sign}"
                        feature = x[:, feature_num] >= threshold if geq else \
                                  x[:, feature_num] > threshold
                        if sign == ">":
                            features.append(feature)
                        else:
                            features.append(~feature)
                    elif len(terms) == 5:
                        feature_num = int(terms[2])
                        leq1 = "=" in terms[1]
                        leq2 = "=" in terms[3]
                        sign1 = terms[1].replace("=", "")
                        sign2 = terms[3].replace("=", "")
                        assert sign1 == "<" and sign2 == "<", f"Invalid signs {sign1, sign2}"
                        threshold1 = float(terms[0])
                        threshold2 = float(terms[4])
                        feature = torch.logical_and(threshold1 <= x[:, feature_num] if leq1 else
                                                    threshold1 < x[:, feature_num],
                                                    x[:, feature_num] <= threshold2 if leq2 else
                                                    x[:, feature_num] < threshold2)

                        features.append(feature)
                    else:
                        raise AttributeError(f"Invalid explanation {terms}")
                else:
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


def predict_minterm(list_of_terms, x):
    if all(isinstance(item, str) for item in list_of_terms):
        features = []
        prev_term = ''
        for feature in list_of_terms:
            if feature.startswith('f'):
                features.append(predict_term(x, prev_term, feature))
            prev_term = feature
        prediction = torch.stack(features, dim=0).prod(dim=0)
        # print(f"({' '.join(lists2)})")
        return f"({' '.join(list_of_terms)})", prediction
    else:
        predictions = []
        prev_term = ''
        for i, item in enumerate(list_of_terms):
            prediction = torch.NoneType
            if isinstance(item, list):
                list_of_terms[i], prediction = predict_minterm(item, x)
            elif item not in ['~', '&']:
                prediction = predict_term(x, prev_term, item)

            if prev_term == '~':
                prediction = ~prediction

            prev_term = item
            if item not in ['~', '&']:
                predictions.append(prediction)
        prediction = torch.stack(predictions, dim=0).prod(dim=0)
        # print(lists2)
        return f"({' '.join(list_of_terms)})", prediction


def predict_term(x, prev_term, term):
    term_id = int(term.split('feature')[1])
    if prev_term == '~':
        return ~x[:, term_id]
    else:
        return x[:, term_id]


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
    :param y: target labels (1D).
    :param x_sample: sample associated to the local formula.
    :param target_class: target class
    :return: Simplified formula
    """
    # # Check if multi class labels
    # if len(y.squeeze().shape) > 1:
    #     y = y.argmax()

    y = to_categorical(y)
    if len(x_sample.shape) == 1:
        x_sample = x_sample.unsqueeze(0)

    y_pred_sample = model((x_sample > 0.5).to(torch.float))
    y_pred_sample = to_categorical(y_pred_sample)
    y_pred_sample = y_pred_sample.view(-1)

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
