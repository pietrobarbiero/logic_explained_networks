import time

import torch

from ..utils.base import NotAvailableError
from ..utils.psi_nn import prune_equal_fanin
from ..logic.psi_nn import generate_fol_explanations
from .base import BaseClassifier, BaseXModel


class PsiNetwork(BaseClassifier, BaseXModel):
    """
        Feed forward Neural Network employing Sigmoid activation function of variable depth completely interpretable.
        After being trained it provides global explanations.

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
                 l1_weight: float = 1e-4, device: torch.device = torch.device('cpu'), name: str = "psi_net.pth",
                 fan_in=2):

        super().__init__(loss, name, device)
        self.n_classes = n_classes
        self.n_features = n_features

        models = torch.nn.ModuleList()
        for j in range(n_classes):
            layers = []
            for i in range(len(hidden_neurons) + 1):
                input_nodes = hidden_neurons[i - 1] if i != 0 else n_features
                output_nodes = hidden_neurons[i] if i != len(hidden_neurons) else 1
                layer = torch.nn.Linear(input_nodes, output_nodes)
                layers.extend([
                    layer,
                    torch.nn.Sigmoid() if i != len(hidden_neurons) else torch.nn.Identity()
                ])
            model = torch.nn.Sequential(*layers)
            models.append(model)

        self.model = models
        self.l1_weight = l1_weight
        self.fan_in = fan_in
        self.need_pruning = True
        self._explanations = None

    def get_loss(self, output: torch.Tensor, target: torch.Tensor, epoch: int = None, epochs: int = None) \
            -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the is employed.
        An L1 weight regularization is also always applied

        :param output: output tensor from the forward function
        :param target: label tensor
        :param epochs:
        :param epoch:
        :return: loss tensor value
        """
        # if epoch is None or epochs is None or epoch + 1 > epochs / 4:
        l1_weight = self.l1_weight
        # else:
        #     l1_weight = self.l1_weight * 4 * (epoch + 1) / epochs
        l1_reg_loss = .0
        if self.need_pruning:
            for model in self.model:
                for layer in model.children():
                    if hasattr(layer, "weight"):
                        l1_reg_loss += torch.norm(layer.weight, 1)
                        l1_reg_loss += torch.norm(layer.bias, 1)
        output_loss = super().get_loss(output, target)
        return output_loss + l1_weight * l1_reg_loss

    def forward(self, x, logits=False):
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after activation in case logits

        :param x: input tensor
        :param logits: whether to return the logits or the probability value after the activation (default)
        :return: output classification
        """
        assert not torch.isnan(x).any(), "Input data contain nan values"
        assert not torch.isinf(x).any(), "Input data contain inf values"
        outputs = []
        for model in self.model:
            output = model(x)
            outputs.append(output)
        output = torch.hstack(outputs)
        if logits:
            return output
        output = self.activation(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        return output

    def prune(self):
        for model in self.model:
            prune_equal_fanin(model, self.fan_in, validate=True, device=self.get_device())

    def get_local_explanation(self, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor, target_class,
                              simplify: bool = True, concept_names: list = None) -> str:
        raise NotAvailableError()

    def get_global_explanation(self, target_class: int, concept_names: list = None, simplify: bool = True,
                               return_time: bool = False, **kwargs):
        """
        Generate explanations.

        :param target_class:
        :param concept_names:
        :param simplify:
        :param return_time:
        :return: Explanation
        """
        start_time = time.time()
        model = self.model[target_class]
        explanations = generate_fol_explanations(model, self.get_device(), concept_names, simplify=True)

        if len(explanations) > 1:
            explanations = explanations[target_class]
        else:
            explanations = explanations[0]
        if return_time:
            return explanations, time.time() - start_time
        return explanations


if __name__ == "__main__":
    pass
