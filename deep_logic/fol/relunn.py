from typing import Tuple, List
import collections
import copy

import torch
import numpy as np
from sympy import simplify_logic

from .sigmoidnn import _build_truth_table
from ..utils.base import collect_parameters
from ..utils.relunn import get_reduced_model


def explain_semi_local(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                       x_sample: torch.Tensor, concept_names: List = None,
                       device: torch.device = torch.device('cpu')) -> str:
    """
    Generate the FOL formula for a specific input.

    :param model: pytorch model
    :param x: input data (train)
    :param y: input labels (train)
    :param x_sample: input sample
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: local explanation
    """
    reduced_model = get_reduced_model(model, x_sample)
    w, bias = collect_parameters(reduced_model, device)
    weights = copy.deepcopy(w[0][0])

    # if all the weights are zero, then the explanation is always 'False'
    w_abs = np.abs(weights)
    w_max = np.max(w_abs)
    if w_max == 0:
        return 'False'

    # w_bool = (w_abs / w_max) > 0.5
    # sort features by weight importance
    sorted_features = np.argsort(-w_abs)

    # # build explanation
    # x_sample = x_sample.unsqueeze(0)
    # explanation = ''
    # used_features = []
    # for j in sorted_features:
    #     if w_bool[j] < 0.5:
    #         break
    #     if explanation:
    #         explanation += ' & '
    #     explanation += f'feature{j:010}' if x_sample[:, j] > 0.5 else f'~feature{j:010}'
    #     used_features.append(j)

    y_pred_sample = (model((x_sample > 0.5).to(torch.float)) > 0.5).to(torch.float)
    mask = (y != y_pred_sample).squeeze()
    x_validation = torch.cat([x[mask], x_sample]).to(torch.bool)
    y_validation = torch.cat([y[mask], y_pred_sample]).squeeze()
    # accuracy, _ = test_explanation(explanation, x_validation, y_validation)

    # if accuracy < 1:
    y_preds_0 = torch.ones(y_validation.size(0), dtype=torch.bool)
    # x_sample = x_sample > 0.5
    explanation = ''
    for j in sorted_features:
        # if j in used_features:
        #     continue

        if x_sample[:, j] > 0.5:
            y_preds = y_preds_0 * x_validation[:, j]
        else:
            y_preds = y_preds_0 * ~x_validation[:, j]

        if y_preds.eq(y_validation).all().item():
            break

        if explanation:
            explanation += ' & '
        explanation += f'feature{j:010}' if x_sample[:, j] > 0.5 else f'~feature{j:010}'

        y_preds_0 = y_preds

    # # Simplify formula
    # for term in explanation.split(' & '):
    #     explanation_simplified = copy.deepcopy(explanation)
    #
    #     if explanation_simplified.endswith(f'{term}'):
    #         explanation_simplified = explanation_simplified.replace(f' & {term}', '')
    #     else:
    #         explanation_simplified = explanation_simplified.replace(f'{term} & ', '')
    #
    #     if explanation_simplified:
    #         accuracy, _ = test_explanation(explanation_simplified, x_validation, y_validation)
    #         if accuracy == 1:
    #             explanation = copy.deepcopy(explanation_simplified)

    if concept_names:
        explanation = replace_names(explanation, concept_names)

    return explanation


def test_explanation(explanation: str, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, np.ndarray]:
    """
    Test explanation

    :param explanation: formula
    :param x: input data
    :param y: input labels
    :return: Accuracy of the explanation and predictions
    """
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

    if len(y.shape) > 1:
        y = torch.argmax(y, dim=1)

    accuracy = sum(predictions == y.detach().numpy().squeeze()) / len(predictions)
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


def combine_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                               target_class: int, topk_explanations: int = 2, concept_names: List = None,
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
        # get prediction for each sample
        output = model(xi)
        pred_class = torch.argmax(output)
        true_class = torch.argmax(yi)

        # generate local explanation only if the prediction is correct
        if pred_class.eq(true_class).item() and pred_class.eq(target_class).item():
            # local_explanation = explain_semi_local(model, x, y, xi,
            #                                        concept_names=None, device=device)
            # local_explanations.append(local_explanation)
            # local_explanation_translated = explain_semi_local(model, x, y, xi,
            #                                                   concept_names=concept_names, device=device)
            # local_explanations_translated.append(local_explanation_translated)
            local_explanation = explain_local(model, x, y, xi, concept_names=None, device=device)
            local_explanations.append(local_explanation)
            local_explanation_translated = explain_local(model, x, y, xi, concept_names=concept_names, device=device)
            local_explanations_translated.append(local_explanation_translated)

    if len(local_explanations) == 0:
        return None, None, None

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
        return None, None, None

    # predictions based on FOL formula
    accuracy, predictions = test_explanation(global_explanation_simplified, x, y)

    # replace concept names
    if concept_names:
        global_explanation_simplified_str = replace_names(global_explanation_simplified_str, concept_names)

    return global_explanation_simplified_str, predictions, counter_translated


def explain_local(model: torch.nn.Module, x, y, x_sample,
                  concept_names: List = None, device: torch.device = torch.device('cpu')):
    x_sample = x_sample.unsqueeze(0)
    y_pred_sample = model(x_sample)
    pred_class = torch.argmax(y_pred_sample, dim=1).item()
    n_classes = len(torch.unique(y))

    w, b = collect_parameters(model, device)
    feature_weights = w[0]
    block_size = feature_weights.shape[0] // n_classes
    feature_used_bool = np.sum(np.abs(feature_weights[pred_class*block_size:(pred_class+1)*block_size]), axis=0) > 0
    feature_used = np.nonzero(feature_used_bool)[0]

    explanation = ''
    for j in feature_used:
        if explanation:
            explanation += ' & '
        explanation += f'feature{j:010}' if x_sample[:, j] > 0.5 else f'~feature{j:010}'

    # # Simplify formula
    # y_pred_sample = (model((x_sample > 0.5).to(torch.float)) > 0.5).to(torch.float)
    # mask = (y != y_pred_sample).squeeze()
    # x_validation = torch.cat([x[mask], x_sample]).to(torch.bool)
    # y_validation = torch.cat([y[mask], y_pred_sample]).squeeze()
    # for term in explanation.split(' & '):
    #     explanation_simplified = copy.deepcopy(explanation)
    #
    #     if explanation_simplified.endswith(f'{term}'):
    #         explanation_simplified = explanation_simplified.replace(f' & {term}', '')
    #     else:
    #         explanation_simplified = explanation_simplified.replace(f'{term} & ', '')
    #
    #     if explanation_simplified:
    #         accuracy, _ = test_explanation(explanation_simplified, x_validation, y_validation)
    #         if accuracy == 1:
    #             explanation = copy.deepcopy(explanation_simplified)

    if concept_names:
        explanation = replace_names(explanation, concept_names)

    return explanation
