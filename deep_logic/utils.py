from typing import Tuple, List

import torch
import numpy as np


def collect_parameters(model: torch.nn.Module,
                       device: torch.device = torch.device('cpu')) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Collect network parameters in two lists of numpy arrays.

    :param model: pytorch model
    :param device: cpu or cuda device
    :return: list of weights and list of biases
    """
    weights, bias = [], []
    for module in model.children():
        if isinstance(module, torch.nn.Linear):
            if device.type == 'cpu':
                weights.append(module.weight.detach().numpy())
                try:
                    bias.append(module.bias.detach().numpy())
                except:
                    pass

            else:
                weights.append(module.weight.cpu().detach().numpy())
                try:
                    bias.append(module.bias.cpu().detach().numpy())
                except:
                    pass

    return weights, bias


def validate_data(x: torch.Tensor) -> None:
    """
    Check that input domain is in [0, 1]

    :param x: input data
    :return:
    """
    assert x.min() >= 0
    assert x.max() <= 1
    return


def validate_network(model: torch.nn.Module, model_type: str = 'relu') -> None:
    """
    Check that the model is a valid deep-logic network.

    :param model: pytorch model
    :return:
    """
    if model_type == 'relu':
        # count number of layers
        n_layers = 0
        for _ in model.children():
            n_layers += 1

        for i, module in enumerate(model.children()):
            if i < n_layers-1:
                assert isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ReLU)

    if model_type == 'psi':
        for module in model.children():
            assert isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Sigmoid)

    return
