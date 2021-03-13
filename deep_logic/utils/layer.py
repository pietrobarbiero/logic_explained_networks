import torch
from ..nn.logic import XLogic, XLogicConv2d


def prune_logic_layers(model: torch.nn.Module, fan_in: int = None,
                       device: torch.device = torch.device('cpu')) -> torch.nn.Module:
    """
    Prune the inputs of the model.

    :param model: pytorch model
    :param fan_in: number of features to retain
    :param device: cpu or cuda device
    :return: pruned model
    """
    model.eval()
    for i, module in enumerate(model.children()):
        # prune only Linear layers
        if isinstance(module, XLogic) or isinstance(module, XLogicConv2d):
            if not module.top:
                _prune(module, fan_in, device)
                # _prune(module.razor_out, fan_in, device, module.razor_in.weight_mask.mean(dim=0))
        # break

    model.train()
    return model


def _prune(module: torch.nn.Module, fan_in: int, device, w = None):
    # pruning
    w_size = (module.weight.shape[0], module.weight.shape[1])

    # identify weights with the lowest absolute values
    w_abs = torch.norm(module.weight, dim=1)
    if w is not None:
        w_abs *= w

    w_sorted = torch.argsort(w_abs, descending=True)

    if fan_in:
        w_to_prune = w_sorted[fan_in:]
    else:
        w_max = torch.max(w_abs)
        w_to_prune = (w_abs / w_max) < 0.5

    mask = torch.ones(w_size)
    # mask[:, w_to_prune] = 0
    mask[w_to_prune, :] = 0

    # prune
    torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=mask.to(device))
    # torch.nn.utils.prune.custom_from_mask(module, name="bias", mask=mask.mean(dim=0).to(device))
    return
