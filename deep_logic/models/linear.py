import torch

from .base import BaseXClassifier


class LogisticRegressionClassifier(BaseXClassifier):
    """
        Logistic Regression class module. It does not provides for explanations.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
        :param loss: torch.nn.modules.loss
            type of loss to employ
     """

    def __init__(self, n_classes: int, n_features: int, loss: torch.nn.modules.loss,
                 device: torch.device = torch.device('cpu'), name: str = "net"):

        super().__init__(name, device)
        self.n_classes = n_classes
        self.n_features = n_features

        layers = [
            torch.nn.Linear(n_features, n_classes),
            torch.nn.Sigmoid(),
        ]
        self.model = torch.nn.Sequential(*layers)
        self.loss = loss

    def forward(self, x) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        super(LogisticRegressionClassifier, self).forward(x)
        output = self.model(x)
        return output

    def get_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the LogisticRegression is
        employed.

        :param output: output tensor from the forward function
        :param target: label tensor
        :return: loss tensor value
        """
        output_loss = self.loss(output, target)
        return output_loss


if __name__ == "__main__":
    pass
