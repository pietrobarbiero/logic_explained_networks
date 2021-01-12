from typing import Tuple

import torch
import numpy as np
from sympy import simplify_logic

from ..utils.base import collect_parameters
from ..utils.relunn import get_reduced_model


def generate_local_explanations(model: torch.nn.Module, x_sample: torch.Tensor, k: int = 5,
                                device: torch.device = torch.device('cpu')) -> str:
    """
    Generate the FOL formula for a specific input.

    :param model: pytorch model
    :param x_sample: input sample
    :param k: upper bound to the number of symbols involved in the explanation (it controls the complexity of the explanation)
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

            if xij >= 0.5:
                explanation += f'f{j}'
            else:
                explanation += f'~f{j}'

    return explanation


def combine_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                               k: int = 5, device: torch.device = torch.device('cpu')) -> Tuple[str, np.array]:
    """
    Generate a global explanation combining local explanations.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param k: upper bound to the number of symbols involved in the explanation (it controls the complexity of the explanation)
    :param device: cpu or cuda device
    :return: global explanation and predictions
    """
    local_explanations = []
    # TODO: multi class
    for xi, yi in zip(x, y):
        # get reduced model for each sample
        model_reduced = get_reduced_model(model, xi)
        output = model_reduced(xi)

        # generate local explanation only if the prediction is correct
        if output > 0.5 and (output > 0.5) == yi:
            local_explanation = generate_local_explanations(model_reduced, xi, k, device)
            local_explanations.append(local_explanation)

    # the global explanation is the disjunction of local explanations
    global_explanation = ' | '.join(local_explanations)
    global_explanation_simplified = simplify_logic(global_explanation, 'dnf', force=True)

    # predictions based on FOL formula
    minterms = str(global_explanation_simplified).split(' | ')
    x_bool = x.detach().numpy() > 0.5
    predictions = np.zeros(x.shape[0], dtype=bool)
    for minterm in minterms:
        minterm = minterm.replace('(', '').replace(')', '').split(' & ')
        local_predictions = np.ones(x.shape[0], dtype=bool)
        for terms in minterm:
            terms = terms.split('f')
            if terms[0] == '~':
                local_predictions *= ~x_bool[:, int(terms[1])]
            else:
                local_predictions *= x_bool[:, int(terms[1])]

        predictions += local_predictions

    return global_explanation_simplified, predictions
