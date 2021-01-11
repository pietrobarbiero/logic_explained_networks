from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from sympy import simplify_logic

import utils
from classifier import Classifier


class InterpretableReLU(Classifier):
    """
        Feed forward Neural Network employing ReLU activation function of variable depth but completely interpretable.
        After being trained it provides for local explanation for the prediction on a single example and global
        explanations on the overall dataset

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
        :param hidden_neurons: list
            number of hidden neurons per layer. The length of the list corresponds to the depth of the network.
        :param loss: torch.nn.modules.loss
            type of loss to employ
        :param l1_weight: float
            weight of the l1 regularization on the weights of the network. Allows extracting compact explanations
     """

    def __init__(self, n_classes: int, n_features: int, hidden_neurons: list, loss: torch.nn.modules.loss,
                 l1_weight: float = 1e-4, device: torch.device = torch.device('cpu'), name: str = "net"):

        self.n_classes = n_classes
        self.n_features = n_features

        layers = []
        for i in range(len(hidden_neurons) + 1):
            input_nodes = hidden_neurons[i-1] if i != 0 else n_features
            output_nodes = hidden_neurons[i] if i != len(hidden_neurons) else n_classes
            layers.extend([
                torch.nn.Linear(input_nodes, output_nodes),
                torch.nn.ReLU() if i != len(hidden_neurons) else torch.nn.Sigmoid()
            ])
        self.model = torch.nn.Sequential(*layers)
        self.loss = loss
        self.l1_weight = l1_weight

        super(InterpretableReLU, self).__init__(name, device)

    def forward(self, x) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        super(InterpretableReLU, self).forward(x)
        output = self.model(x)
        return output

    def get_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the InterpretableReLU is
        employed. An L1 weight regularization is also always applied

        :param output: output tensor from the forward function
        :param target: label tensor
        :return: loss tensor value
        """
        l1_reg_loss = .0
        for layer in self.model.children():
            if hasattr(layer, "weight"):
                l1_reg_loss += torch.sum(torch.abs(layer.weight))
        output_loss = self.loss(output, target)
        return output_loss + l1_reg_loss

    def _get_reduced_model(self, x_sample: torch.Tensor) -> torch.nn.Module:
        """
        Get 1-layer model corresponding to the firing path of the model for a specific sample.

        :param self: InterpretableReLU
        :param x_sample: input sample
        :return: reduced model
        """
        x_sample_copy = deepcopy(x_sample)

        n_linear_layers = 0
        for i, module in enumerate(self.children()):
            if isinstance(module, torch.nn.Linear):
                n_linear_layers += 1

        # compute firing path
        count_linear_layers = 0
        weights_reduced = None
        bias_reduced = None
        for i, module in enumerate(self.children()):
            if isinstance(module, torch.nn.Linear):
                weight = deepcopy(module.weight).detach()
                bias = deepcopy(module.bias).detach()

                # linear layer
                hi = module(x_sample_copy)
                # relu activation
                ai = torch.relu(hi)

                # prune nodes that are not firing
                # (except for last layer where we don't have a relu!)
                if count_linear_layers != n_linear_layers - 1:
                    weight[hi <= 0] = 0
                    bias[hi <= 0] = 0

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

    def generate_local_explanations(self, x_sample: torch.Tensor, k: int = 5,
                                    device: torch.device = torch.device('cpu')) -> str:
        """
        Generate the FOL formula for a specific input.

        :param self: pytorch model
        :param x_sample: input sample
        :param k: upper bound to the number of symbols involved in the explanation
            (it controls the complexity of the explanation)
        :param device: cpu or cuda device
        :return: local explanation
        """
        model_reduced = self.get_reduced_model(x_sample)

        x_sample_np = x_sample.detach().numpy()
        weights, _ = utils.collect_parameters(model_reduced, device)
        weights = weights[0][0]

        # normalize weights
        w_abs = np.abs(weights)
        w_max = np.max(w_abs)
        if w_max > 0:
            w_bool = (w_abs / w_max) > 0.5

            # if the explanation is too complex,
            # reduce the number of symbols to the k most relevant
            if sum(w_bool) > k:
                w_sorted = np.argsort(-w_abs)[:k]
                w_bool = np.zeros(w_bool.shape)
                w_bool[w_sorted] = 1
        else:
            return 'False'

        # build explanation
        explanation = ''
        for j, (wj, xij) in enumerate(zip(w_bool, x_sample_np)):
            if wj:
                if explanation:
                    explanation += ' & '

                if xij >= 0.5:
                    explanation += f'f{j}'
                else:
                    explanation += f'~f{j}'

        return explanation

    def combine_local_explanations(self, x: torch.Tensor, y: torch.Tensor,
                                   k: int = 5) -> Tuple[str, np.array]:
        """
        Generate a global explanation combining local explanations.

        :param self: pytorch model
        :param x: input samples
        :param y: target labels
        :param k: upper bound to the number of symbols involved in the explanation (it controls the complexity of the
        explanation)
        :return: global explanation and predictions
        """
        local_explanations = []
        for xi, yi in zip(x, y):
            # get reduced model for each sample
            output = self(xi)

            # generate local explanation only if the prediction is correct
            if output > 0.5 and (output > 0.5) == yi:
                local_explanation = self.generate_local_explanations(xi, k)
                local_explanations.append(local_explanation)

        # the global explanation is the disjunction of local explanations
        global_explanation = ' | '.join(local_explanations)
        global_explanation_simplified = simplify_logic(global_explanation, 'dnf')

        # predictions based on FOL formula
        minterms = str(global_explanation_simplified).split(' | ')
        x_bool = x.detach().numpy() > 0.5
        predictions = np.zeros(x.shape[0], dtype=bool)
        for minterm in minterms:
            minterm = minterm.replace('(', '').replace(')', '').split(' & ')
            local_predictions = np.ones(x.shape[0], dtype=bool)
            for terms in minterm:
                terms = terms.split('f')
                if terms[0] == '~':
                    local_predictions *= ~x_bool[:, int(terms[1])]
                else:
                    local_predictions *= x_bool[:, int(terms[1])]

            predictions += local_predictions

        return global_explanation_simplified, predictions


if __name__ == "__main__":
    pass
