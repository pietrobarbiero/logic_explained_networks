import torch
from torch.nn.utils import prune
from copy import deepcopy


def prune_features(model: torch.nn.Module, n_classes: int,
                   device: torch.device = torch.device('cpu')) -> torch.nn.Module:
    """
    Prune the inputs of the model.

    :param model: pytorch model
    :param n_classes: number of classes to retain
    :param device: cpu or cuda device
    :return: pruned model
    """
    model.eval()
    for i, module in enumerate(model.children()):
        # prune only Linear layers
        if isinstance(module, torch.nn.Linear):
            # pruning
            blocks = []
            block_size = (module.weight.shape[0] // n_classes, module.weight.shape[1])
            for c in range(n_classes):
                # identify weights with the lowest absolute values
                w_abs = torch.norm(module.weight[c*block_size[0]:(c+1)*block_size[0]], dim=0)
                w_max = torch.max(w_abs)
                w_bool = (w_abs / w_max) < 0.5
                mask = torch.ones(block_size)
                mask[:, w_bool] = 0
                blocks.append(mask)
                print(f"Pruned {w_bool.sum()}/{w_bool.shape[0]} features")

            # prune
            final_mask = torch.vstack(blocks).to(device)
            torch.nn.utils.prune.custom_from_mask(module, name="weight", mask=final_mask)

        break

    model.train()
    return model


def get_reduced_model(model: torch.nn.Module, x_sample: torch.Tensor,
                      bias: bool = True, activation: bool = True) -> torch.nn.Module:
    """
    Get 1-layer model corresponding to the firing path of the model for a specific sample.

    :param model: pytorch model
    :param x_sample: input sample
    :param device: cpu or cuda device
    :param bias: True if model has bias
    :param activation: True if you want to add a sigmoid activation on top
    :return: reduced model
    """
    x_sample_copy = deepcopy(x_sample)

    n_linear_layers = 0
    for i, module in enumerate(model.children()):
        if isinstance(module, torch.nn.Linear):
            n_linear_layers += 1

    # compute firing path
    count_linear_layers = 0
    weights_reduced = None
    bias_reduced = None
    b = None
    for i, module in enumerate(model.children()):
        if isinstance(module, torch.nn.Linear):
            weight = deepcopy(module.weight.detach())
            if bias:
                b = deepcopy(module.bias.detach())

            # linear layer
            hi = module(x_sample_copy)
            # relu activation
            ai = torch.relu(hi)

            # prune nodes that are not firing
            # (except for last layer where we don't have a relu!)
            if count_linear_layers != n_linear_layers - 1:
                weight[hi <= 0] = 0
                if bias:
                    b[hi <= 0] = 0

            # compute reduced weight matrix
            if i == 0:
                weights_reduced = weight
                if bias:
                    bias_reduced = b
            else:
                weights_reduced = torch.matmul(weight, weights_reduced)
                if bias:
                    bias_reduced = torch.matmul(weight, bias_reduced) + bias

            # the next layer will have the output of the current layer as input
            x_sample_copy = ai
            count_linear_layers += 1

    # build reduced network
    linear = torch.nn.Linear(weights_reduced.shape[1],
                             weights_reduced.shape[0])
    state_dict = linear.state_dict()
    state_dict['weight'].copy_(weights_reduced.clone().detach())
    if bias:
        state_dict['bias'].copy_(bias_reduced.clone().detach())

    layers = [linear]
    if activation:
        layers.append(torch.nn.Sigmoid())

    model_reduced = torch.nn.Sequential(*layers)
    model_reduced.eval()

    return model_reduced
