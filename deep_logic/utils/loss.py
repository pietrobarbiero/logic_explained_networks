import torch

eps = 1e-10


class MixedMultiLabelLoss(torch.nn.modules.loss._Loss):
    def __init__(self, exclusive_classes_mask: torch.tensor, excl_loss=torch.nn.CrossEntropyLoss(),
                 non_excl_loss=torch.nn.BCEWithLogitsLoss()):
        super(MixedMultiLabelLoss, self).__init__()
        assert exclusive_classes_mask.dtype == torch.bool, "Only boolean mask are allowed"
        self.exclusive_classes = exclusive_classes_mask
        self.excl_loss = excl_loss
        self.non_excl_loss = non_excl_loss

    def __call__(self, output, target, *args, **kwargs) -> torch.tensor:
        assert output.shape[1] == self.exclusive_classes.squeeze().shape[0], \
            f"boolean mask shape {self.exclusive_classes.squeeze().shape}, " \
            f"different from output number of classes {output.shape[1]}"
        excl_output = output[:, self.exclusive_classes]
        excl_target = target[:, self.exclusive_classes]
        excl_target = excl_target.argmax(dim=1)
        non_excl_output = output[:, ~self.exclusive_classes]
        non_excl_target = target[:, ~self.exclusive_classes]
        excl_loss = self.excl_loss(excl_output, excl_target)
        non_excl_loss = self.non_excl_loss(non_excl_output, non_excl_target)
        return excl_loss + non_excl_loss


class MutualInformationLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def __call__(self, output, *args, **kwargs) -> torch.tensor:
        output_probability = torch.nn.Sigmoid()(output)
        return 1 - mutual_information(output_probability, normalized=True)


def _conditional_probabilities(x):
    # Normalized probability over all the outputs on each sample that each outputs holds true
    z = 0.99
    d = torch.as_tensor(x.shape[1])
    beta = torch.log(z * (d - 1) / (1 - z))
    sum_outputs_over_sample = torch.reshape(torch.sum(torch.exp(beta * x), 1), (x.shape[0], 1))
    cond_probabilities = torch.div(torch.exp(beta * x), sum_outputs_over_sample + eps) + eps

    return torch.squeeze(cond_probabilities)


def _conditional_entropy(output, sample_probability):
    # Compute the conditional entropy by summing over all the outputs and over all samples the product of...
    cond_probabilities = _conditional_probabilities(output)
    log_cond_prob = torch.log(cond_probabilities)
    c_entropy_on_sample = torch.sum(torch.multiply(cond_probabilities, log_cond_prob), 1)
    cond_entropy = - torch.sum(c_entropy_on_sample * torch.squeeze(sample_probability), 0)
    cond_entropy = torch.squeeze(cond_entropy)

    return cond_entropy


def _entropy(output, sample_probability):
    # Compute the marginal_probabilities of each output by summing on all the samples the cond_probabilities
    cond_probabilities = _conditional_probabilities(output) * sample_probability
    marginal_probabilities = torch.sum(cond_probabilities, 0)
    marginal_probabilities = torch.reshape(marginal_probabilities, (1, output.shape[1]))

    # Compute the entropy by summing on all the outputs the product among the marginal_probabilities and their log
    entropy = - torch.sum(torch.multiply(marginal_probabilities, torch.log(marginal_probabilities)), 1)

    return torch.squeeze(entropy)


def mutual_information(output: torch.Tensor, sample_probability=None, normalized=False) -> torch.tensor:
    # Sample probability: if not given may be supposed to be = 1/n_sample.
    # Anyway need to be normalized to sum(p(xi))= 1
    if sample_probability is None:
        n_samples = torch.as_tensor(output.shape[0])
        sample_probability = 1 / n_samples
    else:
        assert sample_probability.shape.ndims == 1, "Wrong sample_probability. Should be an array (n_sample, 1), " \
                                                    "received an array with shape " + sample_probability.shape
        sample_probability = sample_probability / (torch.sum(sample_probability) + eps) + eps
        sample_probability = torch.reshape(sample_probability, shape=(sample_probability.shape[0], 1))

    entropy_t = _entropy(output, sample_probability)
    cond_entropy_t = _conditional_entropy(output, sample_probability)
    mutual_info_t = entropy_t - cond_entropy_t

    if normalized:
        return mutual_info_t / entropy_t

    return mutual_info_t

