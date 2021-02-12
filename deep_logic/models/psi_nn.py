import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..logic.base import replace_names
from ..utils.base import NotAvailableError
from ..utils.psi_nn import prune_equal_fanin, validate_network, validate_pruning
from ..logic.psi_nn import generate_fol_explanations
from .base import BaseClassifier, BaseXModel
from ..utils.metrics import Metric, TopkAccuracy, Accuracy


class PsiNetwork(BaseClassifier, BaseXModel):
    """
        Feed forward Neural Network employing Sigmoid activation function of variable depth but completely interpretable.
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
                 l1_weight: float = 1e-4, device: torch.device = torch.device('cpu'), name: str = "net"):

        super().__init__(name, device)
        self.n_classes = n_classes
        self.n_features = n_features

        layers = []
        for i in range(len(hidden_neurons) + 1):
            input_nodes = hidden_neurons[i - 1] if i != 0 else n_features
            output_nodes = hidden_neurons[i] if i != len(hidden_neurons) else n_classes
            layers.extend([
                torch.nn.Linear(input_nodes, output_nodes),
                torch.nn.Sigmoid()
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
        forward method extended from Classifier. Here input data goes through the layer of the Sigmoid network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        super(PsiNetwork, self).forward(x)
        output = self.model(x)
        return output

    def prune(self, fan_in: int):
        prune_equal_fanin(self.model, fan_in, validate=True, device=self.get_device())

    def get_local_explanation(self, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor, target_class,
                              simplify: bool = True, concept_names: list = None) -> str:
        raise NotAvailableError()

    def get_global_explanation(self, target_class: int, concept_names: list = None, **kwargs):
        """
        Generate explanations.

        :param target_class:
        :param concept_names:
        :return: Explanation
        """
        explanations = generate_fol_explanations(self.model, self.get_device())
        if len(explanations) > 1:
            explanations = explanations[target_class]
        else:
            explanations = explanations[0]
        if concept_names is not None:
            explanations = replace_names(explanations, concept_names)
        return explanations

    def fit(self, train_set: Dataset, val_set: Dataset, batch_size: int = 32, epochs: int = 10, num_workers: int = 0,
            l_r: float = 0.1, metric: Metric = TopkAccuracy(), device: torch.device = torch.device("cpu"),
            verbose: bool = True, fanin: int = 2) -> pd.DataFrame:
        """
        fit function that execute many of the common operation generally performed by many method during training.
        Adam optimizer is always employed

        :param train_set: training set on which to train
        :param val_set: validation set used for early stopping
        :param batch_size: number of training data for each step of the training
        :param epochs: number of epochs to train the model
        :param num_workers: number of process to employ to fetch data
        :param l_r: learning rate parameter of the Adam optimizer
        :param metric: metric to evaluate the predictions of the network
        :param device: device on which to perform the training
        :param verbose: whether to output or not epoch metrics
        :param fanin: fan-in of neurons after pruning
        :return: pandas dataframe collecting the metrics from each epoch
        """

        # Setting device
        device = torch.device(device)
        self.to(device), self.train()

        # Setting loss function and optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=l_r)

        # Training epochs
        best_acc, best_epoch = 0.0, 0
        train_accs, val_accs, tot_losses = [], [], []
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=True)
        pbar = tqdm(range(epochs), ncols=100, position=0, leave=True) if verbose else None
        torch.autograd.set_detect_anomaly(True)
        need_pruning = True
        for epoch in range(epochs):
            tot_losses_i = []
            train_outputs, train_labels = [], []
            for data in train_loader:
                # Load batch (dataset, labels) on the correct device
                batch_data, batch_labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # Network outputs on the current batch of dataset
                batch_outputs = self.forward(batch_data)

                # Compute losses and update gradients
                tot_loss = self.get_loss(batch_outputs, batch_labels)
                tot_loss.backward()
                optimizer.step()

                # Data moved to cpu again
                batch_outputs, batch_labels = batch_outputs.detach().cpu(), batch_labels.detach().cpu()
                tot_loss = tot_loss.detach().cpu()
                train_outputs.append(batch_outputs), train_labels.append(batch_labels), tot_losses_i.append(tot_loss)

            train_outputs, train_labels = torch.cat(train_outputs), torch.cat(train_labels)
            tot_losses_i = torch.stack(tot_losses_i)

            # Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
            train_acc = self.evaluate(train_set, batch_size, metric, num_workers, device, train_outputs, train_labels)
            val_acc = self.evaluate(val_set, batch_size, metric, device=device)

            # Save epoch results
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            tot_losses.append(tot_losses_i.mean().item())

            if verbose:
                pbar.set_postfix({"Train_acc": f"{train_acc:.1f}", "Val_acc": f"{val_acc:.1f}",
                                  "best_epoch": best_epoch})
                pbar.update()

            # Prune network
            if epoch >= epochs // 2 and need_pruning:
                self.prune(fanin)
                need_pruning = False

            # Save best model
            if val_acc >= best_acc and epoch > epochs // 2 or epochs == 1:
                best_acc = val_acc
                best_epoch = epoch + 1
                self.save()

        # Best model is loaded and saved again with buffer "trained" set to true
        self.load(device, set_trained=True)

        # Performance dictionary
        performance_dict = {
            "Tot losses": tot_losses,
            "Train accs": train_accs,
            "Val accs": val_accs,
            "Best epoch": best_epoch,
        }
        performance_df = pd.DataFrame(performance_dict)
        return performance_df


if __name__ == "__main__":
    pass
