import torch

from ..nn import XLinear
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
                 l1_weight: float = 1e-4, device: torch.device = torch.device('cpu'), name: str = "psi_net", fan_in=2):

        super().__init__(loss, name, device)
        self.n_classes = n_classes
        self.n_features = n_features

        layers = []
        for i in range(len(hidden_neurons) + 1):
            input_nodes = hidden_neurons[i - 1] if i != 0 else n_features
            output_nodes = hidden_neurons[i] if i != len(hidden_neurons) else n_classes
            if i == 0:
                layer = torch.nn.Linear(input_nodes, output_nodes * self.n_classes)
            elif i != len(hidden_neurons):
                layer = XLinear(input_nodes, output_nodes, self.n_classes)
            else:
                layer = XLinear(input_nodes, 1, self.n_classes)
            layers.extend([
                layer,
                torch.nn.Sigmoid() if i != len(hidden_neurons) else torch.nn.Identity()
            ])
        self.model = torch.nn.Sequential(*layers)
        self.l1_weight = l1_weight
        self.fan_in = fan_in
        self.need_pruning = True

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
        output_loss = super().get_loss(output, target)
        return output_loss + self.l1_weight * l1_reg_loss

    def prune(self):
        prune_equal_fanin(self.model, self.fan_in, validate=True, device=self.get_device())

    def get_local_explanation(self, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor, target_class,
                              simplify: bool = True, concept_names: list = None) -> str:
        raise NotAvailableError()

    def get_global_explanation(self, target_class: int, concept_names: list = None, simplify: bool = True, **kwargs):
        """
        Generate explanations.

        :param target_class:
        :param concept_names:
        :param simplify:
        :return: Explanation
        """
        explanations = generate_fol_explanations(self.model, self.get_device(), concept_names, simplify=True)
        if len(explanations) > 1:
            explanations = explanations[target_class]
        else:
            explanations = explanations[0]
        return explanations


if __name__ == "__main__":
    pass
