from abc import abstractmethod
from typing import Tuple

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from deep_logic.utils.base import ClassifierNotTrainedError, IncompatibleClassifierError
from deep_logic.utils.metrics import Metric, TopkAccuracy, Accuracy


class BaseXModel:
    """Base class for all models in XDL."""

    @abstractmethod
    def explain(self, x: torch.Tensor):
        """
        Get explanation of model decision taken on the input x.

        :param x: input tensor
        :return: Explanation.
        """
        pass


class BaseClassifier(torch.nn.Module):
    """
    Classifier is an abstract class representing a generic classifier. It already provides for a set of common
    methods such as the fit(), the save() and the load() functions.
    init(), forward() and get_loss() methods are required to be implemented by extending classes

    :param name: str
        name of the network: used for saving and loading purposes
    :param device: torch.device
        device on which to load the model after instantiating
    :var trained: torch.bool
        flag set at the end of the training and saved with the model. Only trained model can be loaded from memory
     """

    @abstractmethod
    def __init__(self, name: str = "net", device: torch.device = torch.device("cpu")):

        super(BaseClassifier, self).__init__()
        self.name = name
        self.register_buffer("trained", torch.tensor(True))
        self.to(device)

    @abstractmethod
    def get_loss(self, output: torch.Tensor, target: torch.Tensor):
        """
        get_loss method is used by each class to calculate its own loss according to the different training strategy

        :param output: output tensor from the forward function
        :param target: label tensor
        """
        pass

    def get_device(self) -> torch.device:
        """
        Return the device on which the classifier is actually loaded

        :return: device in use
        """
        return [*self.model.parameters()][0].device

    @abstractmethod
    def forward(self, x):
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        assert not torch.isnan(x).any(), "Input data contain nan values"
        assert not torch.isinf(x).any(), "Input data contain inf values"

    def fit(self, train_set: Dataset, val_set: Dataset, batch_size: int = 32, epochs: int = 10, num_workers: int = 0,
            l_r: float = 0.01, lr_scheduler: bool = False, metric: Metric = TopkAccuracy(), device: torch.device = torch.device("cpu"),
            verbose: bool = True) -> pd.DataFrame:
        """
        fit function that execute many of the common operation generally performed by many method during training.
        Adam optimizer is always employed

        :param train_set: training set on which to train
        :param val_set: validation set used for early stopping
        :param batch_size: number of training data for each step of the training
        :param epochs: number of epochs to train the model
        :param num_workers: number of process to employ when loading data
        :param l_r: learning rate parameter of the Adam optimizer
        :param lr_scheduler: whether to use learning rate scheduler (ReduceLROnPleteau)
        :param metric: metric to evaluate the predictions of the network
        :param device: device on which to perform the training
        :param verbose: whether to output or not epoch metrics
        :return: pandas dataframe collecting the metrics from each epoch
        """

        # Setting device
        self.to(device), self.train()

        # Setting loss function and optimizer
        parameters = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=l_r, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=verbose, factor=0.33, min_lr=1e-3*l_r,
                                      patience=epochs//10) if lr_scheduler else None

        # Training epochs
        best_acc, best_epoch = 0.0, 0
        train_accs, val_accs, tot_losses = [], [], []
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=True,
                                                   num_workers=num_workers, prefetch_factor=4 if num_workers != 0 else 2)
        pbar = tqdm(range(epochs), ncols=100, position=0, leave=True) if verbose else None
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            tot_losses_i = []
            train_outputs, train_labels = [], []
            for i, data in enumerate(train_loader):
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
            val_acc = self.evaluate(val_set, batch_size, metric, num_workers, device)

            # Step learning rate scheduler
            if lr_scheduler:
                scheduler.step(train_acc)

            # Save epoch results
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            tot_losses.append(tot_losses_i.mean().item())

            if verbose:
                pbar.set_postfix({"Tr_acc": f"{train_acc:.1f}", "Val_acc": f"{val_acc:.1f}",
                                  "Loss": f"{tot_losses[-1]:.3f}", "best_e": best_epoch})
                pbar.update()
                print(" ")

            # Save best model
            if val_acc >= best_acc and epoch >= epochs / 2 or epochs == 1:
                best_acc = val_acc
                best_epoch = epoch + 1
                self.save()

        if verbose:
            pbar.close()

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

    def evaluate(self, dataset: Dataset, batch_size: int = 64,
                 metric: Metric = Accuracy(), num_workers: int = 0,
                 device: torch.device = torch.device("cpu"), outputs=None, labels=None) -> float:
        """
        Evaluate function to test the performance of the model on a certain dataset without training

        :param dataset: dataset on which to test
        :param batch_size: number of training data for each step of the training
        :param metric: metric to evaluate the predictions of the network
        :param num_workers: number of process to employ when loading data
        :param device: device on which to perform the training
        :param outputs: if the output is passed is not calculated again
        :param labels: to be passed together with outputs
        :return: metric evaluated on the dataset
        """
        self.eval(), self.to(device)
        with torch.no_grad():
            if outputs is None or labels is None:
                outputs, labels = self.predict(dataset, batch_size, num_workers,  device)
            metric_val = metric(outputs, labels)
        self.train()
        return metric_val

    def predict(self, dataset, batch_size: int = 64, num_workers: int = 4,
                device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict function to compute the prediction of the model on a certain dataset

        :param dataset: dataset on which to test
        :param batch_size: number of training data for each step of the training
        :param num_workers: number of process to employ when loading data
        :param device: device on which to perform the training
        :return: a tuple containing the outputs computed on the dataset and the labels
        """
        outputs, labels = [], []
        loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)
        for i, data in enumerate(loader):
            batch_data, batch_labels, = data[0].to(device), data[1].to(device)
            batch_outputs = self.forward(batch_data)
            outputs.append(batch_outputs)
            labels.append(batch_labels)
        return torch.cat(outputs), torch.cat(labels)

    def save(self, set_trained=False, name=None) -> None:
        """
        Save model on a file named with the name of the model if parameter name is not set.

        :param set_trained: Used to set the buffer flag "trained" at the end of training.
        :param name: Save the model with a name different from the one assigned in the __init__
        """
        if name is None:
            name = self.name
        if set_trained:
            self._set_trained()
        torch.save(self.state_dict(), name)

    def load(self, device, set_trained=False, name=None) -> None:
        """
        Load model on a specific device (can be different from the one used during training).
        If set_trained is true than the model flag "trained" is set to true first and the model is saved again.
        If set_trained is not set and the model flag "trained" is not true a ClassifierNotTrainedError is raised

        :param device: device on which to load the model
        :param set_trained: whether to set the buffer flag "trained" before loading or not.
        :param name: Load a model with a name different from the one assigned in the __init__
        """
        if name is None:
            name = self.name
        try:
            incompatible_keys = self.load_state_dict(torch.load(name,
                                                                map_location=torch.device(device)), strict=False)
        except FileNotFoundError:
            raise ClassifierNotTrainedError() from None
        else:
            if len(incompatible_keys.missing_keys) > 0 or len(incompatible_keys.unexpected_keys) > 0:
                raise IncompatibleClassifierError(incompatible_keys.missing_keys, incompatible_keys.unexpected_keys)
        if set_trained:
            self.save(set_trained=True)
        if not self.trained:
            raise ClassifierNotTrainedError()

    def _set_trained(self) -> None:
        """
        Internal function used to set the buffer "trained" to true
        """
        self.trained = torch.tensor(True)


if __name__ == "__main__":
    pass
