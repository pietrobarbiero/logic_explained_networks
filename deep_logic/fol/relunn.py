from typing import Tuple, List
import collections

import torch
import numpy as np
from sympy import simplify_logic

from ..utils.base import collect_parameters
from ..utils.relunn import get_reduced_model


def generate_local_explanations(model: torch.nn.Module, x_sample: torch.Tensor, k: int = 5,
                                concept_names: List = None,
                                device: torch.device = torch.device('cpu')) -> str:
    """
    Generate the FOL formula for a specific input.

    :param model: pytorch model
    :param x_sample: input sample
    :param k: upper bound to the number of symbols involved in the explanation (it controls the complexity of the explanation)
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: local explanation
    """
    x_sample_np = x_sample.detach().numpy()
    weights, _ = collect_parameters(model, device)
    weights = weights[0][0]

    # normalize weights
    w_abs = np.abs(weights)
    w_max = np.max(w_abs)
    if w_max > 0:
        w_bool = (w_abs / w_max) > 0.5

        # if the explanation is too complex,
        # reduce the number of symbols to the k most relevant
        if sum(w_bool) > k:
            w_sorted = np.argsort(-w_abs)[:k]
            w_bool = np.zeros(w_bool.shape)
            w_bool[w_sorted] = 1
    else:
        return 'False'

    # build explanation
    explanation = ''
    for j, (wj, xij) in enumerate(zip(w_bool, x_sample_np)):
        if wj:
            if explanation:
                explanation += ' & '

            if concept_names:
                term = concept_names[j]
            else:
                term = f'feature{j:05}'

            if xij >= 0.5:
                explanation += term
            else:
                explanation += f'~{term}'

    return explanation


def combine_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                               max_concepts: int = 5, topk_explanations: int = 2, concept_names: List = None,
                               device: torch.device = torch.device('cpu')) -> Tuple[str, np.array, collections.Counter]:
    """
    Generate a global explanation combining local explanations.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param max_concepts: upper bound to the number of concepts involved in a local explanation (it controls the complexity of the local explanation)
    :param topk_explanations: number of most common local explanations to combine in a global explanation (it controls the complexity of the global explanation)
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :return: Global explanation, predictions, and ranking of local explanations
    """
    local_explanations = []
    local_explanations_translated = []
    # TODO: multi class
    for xi, yi in zip(x, y):
        # get reduced model for each sample
        model_reduced = get_reduced_model(model, xi)
        output = model_reduced(xi)

        # generate local explanation only if the prediction is correct
        if output > 0.5 and (output > 0.5) == yi:
            local_explanation = generate_local_explanations(model_reduced, xi, max_concepts, concept_names=None, device=device)
            local_explanations.append(local_explanation)
            local_explanation_translated = generate_local_explanations(model_reduced, xi, max_concepts, concept_names=concept_names, device=device)
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

    # predictions based on FOL formula
    minterms = str(global_explanation_simplified).split(' | ')
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

    global_explanation_simplified_str = str(global_explanation_simplified)

    if concept_names:
        feature_abbreviations = [f'feature{i:05}' for i in range(len(concept_names))]
        mapping = []
        for f_abbr, f_name in zip(feature_abbreviations, concept_names):
            mapping.append((f_abbr, f_name))

        for k, v in mapping:
            global_explanation_simplified_str = global_explanation_simplified_str.replace(k, v)

    return global_explanation_simplified_str, predictions, counter_translated
