import torch
from torch.nn.utils import prune

from .utils import validate_network


def prune_equal_fanin(model: torch.nn.Module, k: int = 2, validate: bool = True) -> torch.nn.Module:
    """
    Prune the dense layers of the network such that each neuron has the same fan-in.

    :param model: pytorch model
    :param k: fan-in
    :param validate: if True then validate the network after pruning
    :return: pruned model
    """
    model.eval()
    for i, module in enumerate(model.children()):
        # prune only Linear layers
        if isinstance(module, torch.nn.Linear):
            # create mask
            mask = torch.ones(module.weight.shape)
            # identify weights with the lowest absolute values
            param_absneg = -torch.abs(module.weight)
            idx = torch.topk(param_absneg, k=param_absneg.shape[1] - k, dim=1)[1]
            for j in range(len(idx)):
                mask[j, idx[j]] = 0
            # prune
            prune.custom_from_mask(module, name="weight", mask=mask)

    if validate:
        validate_network(model)
        validate_pruning(model)

    return model


def validate_pruning(model: torch.nn.Module):
    """
    Validate pruned network.

    :param model: pytorch model
    :return:
    """
    for module in model.children():
        if isinstance(module, torch.nn.Linear):
            assert (module.weight != 0).sum(dim=1).sum().item() == 2 * module.weight.size(0)
    return
