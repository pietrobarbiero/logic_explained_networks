from typing import Tuple, List

import sklearn
import torch
import numpy as np
from sklearn.tree import _tree, DecisionTreeClassifier


def set_seed(seed):
    """
    Static method used to set the seed for an experiment. Needs to be called before doing anything else.

    :param seed:
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_categorical(y: torch.Tensor) -> torch.Tensor:
    """
    Transform input tensor to categorical.

    :param y: input tensor.
    :return: Categorical tensor
    """
    if len(y.shape) == 2 and y.shape[1] > 1:
        # one hot encoding to categorical
        yc = torch.argmax(y, dim=1)

    else:
        # binary/probabilities to categorical
        yc = y.squeeze() > 0.5

    if len(yc.size()) == 0:
        yc = yc.unsqueeze(0)

    return yc


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
            if i < n_layers - 1:
                assert isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ReLU)

    if model_type == 'psi':
        for module in model.children():
            assert isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Sigmoid)

    return


def tree_to_formula(tree: DecisionTreeClassifier, concept_names: List[str], target_class: int) -> str:
    """
    Translate a decision tree into a set of decision rules.

    :param tree: sklearn decision tree
    :param concept_names: concept names
    :param target_class: target class
    :return: decision rule
    """
    tree_ = tree.tree_
    feature_name = [
        concept_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto = dict()

    global k
    global explanation
    explanation = ''
    k = 0

    def recurse(node, depth, parent):
        global k
        global explanation
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s = f'{name} <= {threshold:.2f}'
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s

            recurse(tree_.children_left[node], depth + 1, node)
            s = f'{name} > {threshold:.2f}'
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent] + ' & ' + s
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k = k + 1
            if tree_.value[node].squeeze().argmax() == target_class:
                explanation += f'({pathto[parent]}) | '

    recurse(0, 1, 0)
    return explanation[:-3]
