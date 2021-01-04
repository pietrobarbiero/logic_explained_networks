import torch
import numpy as np

from ..utils import collect_parameters


def generate_local_explanations(model: torch.nn.Module, x_sample: torch.Tensor, device: torch.device = torch.device('cpu')):
    """
    Generate the FOL formula for a specific input.

    :param model: pytorch model
    :param x_sample: input sample
    :param device: cpu or cuda device
    :return: reduced model
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
