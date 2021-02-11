from typing import List

import torch

from ..utils.relunn import get_reduced_model
from ..logic.relunn import combine_local_explanations
from .base import BaseClassifier, BaseXModel


class XReluClassifier(BaseClassifier, BaseXModel):
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

        super().__init__(name, device)
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
        return output_loss + self.l1_weight * l1_reg_loss

    def forward(self, x) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        super(XReluClassifier, self).forward(x)
        output = self.model(x)
        return output

    def get_reduced_model(self, x_sample: torch.Tensor) -> torch.nn.Module:
        """
        Get 1-layer model corresponding to the firing path of the model for a specific sample.

        :param x_sample: input sample
        :return: reduced model
        """
        self.reduced_model = get_reduced_model(self.model, x_sample)
        return self.reduced_model

    def explain(self, x: torch.Tensor, y: torch.Tensor = None, sample_id: int = None, local: bool = True,
                concept_names: List = None, device: torch.device = torch.device('cpu')):
        """
        Generate explanations.

        :param x: input samples
        :param y: target labels
        :param sample_id: number of the sample to be explained (for local explanation)
        :param local: require local or global explanations
        :param concept_names: concept names to use in the explanation
        :param device: cpu or cuda device
        :return: Explanation
        """
        assert len(x.shape) <= 2, 'Only 1 or 2 dimensional data are allowed.'
        assert sample_id is not None or not local, "Local explanation requires sample_id to be defined"
        if local:
            # if len(x.shape) == 2:
            #     assert x.shape[0] == 1, 'Local explanation requires 1 single sample.'
            #
            raise NotImplementedError()

        else:
            # return combine_local_explanations(self.model, x, y, concept_names, device)
            raise NotImplementedError()


if __name__ == "__main__":
    pass
