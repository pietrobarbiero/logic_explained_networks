import torch
from ..nn.logic import XLogic


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
        if isinstance(module, XLogic):
            if not module.top:
                # pruning
                w_size = (module.weight.shape[0], module.weight.shape[1])

                # identify weights with the lowest absolute values
                w_abs = torch.norm(module.weight, dim=0)
                w_sorted = torch.argsort(w_abs, descending=True)

                if fan_in:
                    w_to_prune = w_sorted[fan_in:]
                else:
                    w_max = torch.max(w_abs)
                    w_to_prune = (w_abs / w_max) < 0.5

                mask = torch.ones(w_size)
                mask[:, w_to_prune] = 0

                # prune
                torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=mask.to(device))

        # break

    model.train()
    return model
