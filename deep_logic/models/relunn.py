import torch

from ..utils.relunn import get_reduced_model
from ..fol.relunn import generate_local_explanations, combine_local_explanations
from .base import BaseXClassifier


class XReluClassifier(BaseXClassifier):
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

    def forward(self, x) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        # super(XReluClassifier, self).forward(x)
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

    def get_reduced_model(self, x_sample: torch.Tensor) -> torch.nn.Module:
        """
        Get 1-layer model corresponding to the firing path of the model for a specific sample.

        :param x_sample: input sample
        :return: reduced model
        """
        self.reduced_model = get_reduced_model(self.model, x_sample)
        return self.reduced_model

    def get_explanation(self, x: torch.Tensor, y: torch.Tensor = None, kind: str = 'local', k: int = 5,
                        device: torch.device = torch.device('cpu')):
        """
        Generate explanations.

        :param x: input samples
        :param y: target labels (required for global explanations
        :param kind: require local or global explanations
        :param k: upper bound to the number of symbols involved in the explanation (it controls the complexity of the
        explanation)
        :param device: cpu or cuda device
        :return: Explanation
        """
        if kind == 'local':
            return generate_local_explanations(self.model, x, k, device)
        elif kind == 'global':
            return combine_local_explanations(self.model, x, y, k, device)


if __name__ == "__main__":
    pass
