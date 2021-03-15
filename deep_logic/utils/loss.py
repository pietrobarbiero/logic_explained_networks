import torch

eps = 1e-10


def _conditional_probabilities(x):
    # Normalized probability over all the constraints on each sample that each constraints holds true
    z = 0.99
    d = torch.FloatTensor(x.shape[1])
    beta = torch.log(z * (d - 1) / (1 - z))
    sum_constraints_over_sample = torch.reshape(torch.sum(torch.exp(beta * x), 1), (x.shape[0], 1))
    cond_probabilities = torch.div(torch.exp(beta * x), sum_constraints_over_sample + eps) + eps

    return torch.squeeze(cond_probabilities)


def _conditional_entropy(output, sample_probability, constraint=None):
    # Compute the conditional entropy by summing over all the constraints and over all samples the product of...
    cond_probabilities = _conditional_probabilities(output)
    log_cond_prob = torch.log(cond_probabilities)
    c_entropy_on_sample = torch.sum(torch.multiply(cond_probabilities, log_cond_prob), 1)
    cond_entropy = - torch.sum(c_entropy_on_sample * torch.squeeze(sample_probability), 0)
    cond_entropy = torch.squeeze(cond_entropy)

    if constraint:
        constraint._cond_prob2 = cond_probabilities
        constraint._log_cond_prob = log_cond_prob
        constraint._cond_entropy = cond_entropy

    return cond_entropy


def _entropy(output, sample_probability, constraint=None):
    # Compute the marginal_probabilities of each constraint by summing on all the samples the cond_probabilities
    cond_probabilities = _conditional_probabilities(output) * sample_probability
    marginal_probabilities = torch.sum(cond_probabilities, 0)
    marginal_probabilities = torch.reshape(marginal_probabilities, (1, output.shape[1].value))

    # Compute the entropy by summing on all the constraints the product among the marginal_probabilities and their log
    entropy = - torch.sum(torch.multiply(marginal_probabilities, torch.log(marginal_probabilities)), 1)

    if constraint:
        constraint._cond_prob1 = cond_probabilities
        constraint._marginal_prob = marginal_probabilities
        constraint._entropy = entropy

    return torch.squeeze(entropy)


def mutual_information(output: torch.Tensor, labels: torch.Tensor = None, sample_probability=None, constraint=None):
    # Sample probability: if not given may be supposed to be = 1/n_sample.
    # Anyway need to be normalized to sum(p(xi))= 1
    if sample_probability is None:
        n_samples = torch.FloatTensor(output.shape[0])
        sample_probability = 1 / n_samples
    else:
        assert sample_probability.shape.ndims == 1, "Wrong sample_probability. Should be an array (n_sample, 1), " \
                                                    "received an array with shape " + sample_probability.shape
        sample_probability = sample_probability / (torch.sum(sample_probability) + eps) + eps
        sample_probability = torch.reshape(sample_probability, shape=(sample_probability.shape[0], 1))

    entropy_t = _entropy(output, sample_probability, constraint=constraint)
    cond_entropy_t = _conditional_entropy(output, sample_probability, constraint=constraint)
    mutual_info_t = entropy_t - cond_entropy_t

    if constraint:
        constraint._sample_probability = sample_probability
        constraint._mutual_info = mutual_info_t

    return mutual_info_t
