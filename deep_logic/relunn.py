import torch
from copy import deepcopy


def get_reduced_model(model: torch.nn.Module, x_sample: torch.Tensor) -> torch.nn.Module:
    """
    Get 1-layer model corresponding to the firing path of the model for a specific sample.

    :param model: pytorch model
    :param x_sample: input sample
    :param device: cpu or cuda device
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
    for i, module in enumerate(model.children()):
        if isinstance(module, torch.nn.Linear):
            weight = deepcopy(module.weight).detach()
            bias = deepcopy(module.bias).detach()

            # linear layer
            hi = module(x_sample_copy)
            # relu activation
            ai = torch.relu(hi)

            # prune nodes that are not firing
            # (except for last layer where we don't have a relu!)
            if count_linear_layers != n_linear_layers-1:
                weight[hi<=0] = 0
                bias[hi<=0] = 0

            # compute reduced weight matrix
            if i == 0:
                weights_reduced = weight
                bias_reduced = bias
            else:
                weights_reduced = torch.matmul(weight, weights_reduced)
                bias_reduced = torch.matmul(weight, bias_reduced) + bias

            # the next layer will have the output of the current layer as input
            x_sample_copy = ai
            count_linear_layers += 1

    # build reduced network
    linear = torch.nn.Linear(weights_reduced.shape[1],
                             weights_reduced.shape[0])
    state_dict = linear.state_dict()
    state_dict['weight'].copy_(weights_reduced.clone().detach())
    state_dict['bias'].copy_(bias_reduced.clone().detach())

    model_reduced = torch.nn.Sequential(*[
        linear,
        torch.nn.Sigmoid()
    ])
    model_reduced.eval()
    return model_reduced
