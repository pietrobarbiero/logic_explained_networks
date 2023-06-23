import torch

from lens.utils.base import NotAvailableError
from .base import BaseClassifier


class BlackBoxClassifier(BaseClassifier):
    """
        BlackBox Neural Network employing ReLU activation function of variable depth but completely interpretable.
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
     """

    def __init__(self, n_classes: int, n_features: int, hidden_neurons: list, loss: torch.nn.modules.loss,
                 device: torch.device = torch.device('cpu'), name: str = "black_box.pth"):

        super().__init__(loss, name, device)
        self.n_classes = n_classes
        self.n_features = n_features
        self.eval_main_classes = False
        self.eval_logits = False

        layers = []
        for i in range(len(hidden_neurons) + 1):
            input_nodes = hidden_neurons[i - 1] if i != 0 else n_features
            output_nodes = hidden_neurons[i] if i != len(hidden_neurons) else n_classes
            layers.extend([
                torch.nn.Linear(input_nodes, output_nodes),
                torch.nn.LeakyReLU() if i != len(hidden_neurons) else torch.nn.Identity()
            ])
        self.model = torch.nn.Sequential(*layers)

    def prune(self):
        raise NotAvailableError("Prune method is not available with BlackBoxClassifier model.")


if __name__ == "__main__":
    pass
