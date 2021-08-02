import torch
from torch.nn.utils import prune

from .base import validate_network


def prune_equal_fanin(model: torch.nn.Module, k: int = 2,
                      validate: bool = True,
                      device: torch.device = torch.device('cpu')) -> torch.nn.Module:
    """
    Prune the dense layers of the network such that each neuron has the same fan-in.

    :param model: pytorch model
    :param k: fan-in
    :param validate: if True then validate the network after pruning
    :param device: cpu or cuda device
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
            mask = mask.to(device)
            prune.custom_from_mask(module, name="weight", mask=mask)
            # print(f"Pruned {k}/{module.weight.shape[1]} weights")

    if validate:
        validate_network(model, 'psi')
        validate_pruning(model, fan_in=k)

    return model


def validate_pruning(model: torch.nn.Module, fan_in: int = 2) -> None:
    """
    Validate pruned network.

    :param model: pytorch model
    :param fan_in: number of feature retained per node
    :return:
    """
    for module in model.children():
        if isinstance(module, torch.nn.Linear):
            assert (module.weight != 0).sum(dim=1).sum().item() == fan_in * module.weight.size(0)
    return
