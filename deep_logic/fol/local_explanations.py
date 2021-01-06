import torch
import numpy as np

from ..utils import collect_parameters
from ..relunn import get_reduced_model


def generate_local_explanations(model: torch.nn.Module, x_sample: torch.Tensor,
                                device: torch.device = torch.device('cpu')) -> str:
    """
    Generate the FOL formula for a specific input.

    :param model: pytorch model
    :param x_sample: input sample
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


def combine_local_explanations(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> str:
    """
    Generate a global explanation combining local explanations.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :return: global explanation
    """
    global_explanation = ''
    for xi, yi in zip(x, y):
        # get reduced model for each sample
        model_reduced = get_reduced_model(model, xi)
        output = model_reduced(xi)

        # generate local explanation only if the prediction is correct
        if output > 0.5 and (output > 0.5) == yi:
            local_explanation = generate_local_explanations(model_reduced, xi)
            # the global explanation is the disjunction of local explanations
            global_explanation += f'{local_explanation} | '

    global_explanation = global_explanation[:-3]
    return global_explanation
