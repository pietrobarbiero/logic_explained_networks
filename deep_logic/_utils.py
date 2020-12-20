from typing import Tuple, List

import torch
import numpy as np


def collect_parameters(model: torch.nn.Module) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Collect network parameters in two lists of numpy arrays.

    :param model: pytorch model
    :return: list of weights and list of biases
    """
    weights, bias = [], []
    for module in model.children():
        if isinstance(module, torch.nn.Linear):
            weights.append(module.weight.detach().numpy())
            bias.append(module.bias.detach().numpy())

    return weights, bias


def validate_data(x: torch.Tensor):
    """
    Check that input domain is in [0, 1]

    :param x: input data
    :return:
    """
    assert x.min() >= 0
    assert x.max() <= 1
    return


def validate_network(model: torch.nn.Module):
    """
    Check that the model is a valid deep-logic network.

    :param model: pytorch model
    :return:
    """
    for module in model.children():
        assert isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Sigmoid)
    return
