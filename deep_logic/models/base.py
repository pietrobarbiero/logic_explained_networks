import os
from abc import abstractmethod, ABC
from typing import Tuple

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from deep_logic.utils.base import ClassifierNotTrainedError, IncompatibleClassifierError
from deep_logic.utils.metrics import Metric, TopkAccuracy, Accuracy
from deep_logic.utils.loss import MutualInformationLoss, MixedMultiLabelLoss


class BaseXModel(ABC):
    """Base class for all models in XDL."""

    @abstractmethod
    def get_local_explanation(self, x: torch.Tensor, y: torch.Tensor, x_sample: torch.Tensor,
                              target_class, simplify: bool = True, concept_names: list = None) -> str:
        """
        Get explanation of model decision taken on the input x_sample.

        :param x: input samples
        :param y: target labels
        :param x_sample: input for which the explanation is required
        :param target_class: class ID
        :param simplify: simplify local explanation
        :param concept_names: list containing the names of the input concepts

        :return: Local Explanation
        """
        pass

    @abstractmethod
    def get_global_explanation(self, x: torch.Tensor, y: torch.Tensor, class_to_explain: int, concept_names: list,
                               *args, **kwargs) -> str:
        """
        Get explanation of model decision taken on the input x.

        :param x: input tensor
        :param y:
        :param concept_names: list of concept names which compose the explanation
        :param class_to_explain: class for which the explanation is given
        :return: The explanation
        """
        pass


class BaseClassifier(torch.nn.Module):
    """
    Classifier is an abstract class representing a generic classifier. It already provides for a set of common
    methods such as the fit(), the save() and the load() functions.
    init(), forward() and get_loss() methods are required to be implemented by extending classes

    :param loss: torch.nn.Module
        loss to employ during training. It may be None when using non-gradient based methods
    :param name: str
        name of the network: used for saving and loading purposes
    :param device: torch.device
        device on which to load the model after instantiating
    :var trained: torch.bool
        flag set at the end of the training and saved with the model. Only trained model can be loaded from memory
     """

    @abstractmethod
    def __init__(self, loss: torch.nn.Module = None, name: str = "net.pth", device: torch.device = torch.device("cpu")):

        super(BaseClassifier, self).__init__()
        if loss is not None:
            assert isinstance(loss, torch.nn.CrossEntropyLoss) or \
                   isinstance(loss, torch.nn.BCEWithLogitsLoss) or \
                   isinstance(loss, MixedMultiLabelLoss) or \
                   isinstance(loss, MutualInformationLoss), \
                   "Only CrossEntropyLoss, BCEWithLogitsLoss, MixedMultiLabelLoss or MutualInformationLoss available."
        self.loss = loss
        self.activation = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1) if isinstance(loss, torch.nn.CrossEntropyLoss) else torch.nn.Sigmoid()
        self.name = name
        self.register_buffer("trained", torch.tensor(False))
        self.need_pruning = False
        self.to(device)

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
        output = self.model(x)
        if logits:
            return output
        output = self.activation(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        return output

    def get_loss(self, output: torch.Tensor, target: torch.Tensor, epoch: int = None,
                 epochs: int = None) -> torch.Tensor:
        """
        get_loss method is used by each class to calculate its own loss according to the different training strategy

        :param output: output tensor from the forward function
        :param target: label tensor
        :param epochs:
        :param epoch:
        """
        if isinstance(self.loss, torch.nn.CrossEntropyLoss):
            target = target.to(torch.long)
            if len(target.squeeze().shape) > 1:
                target = target.argmax(dim=1)
            loss = self.loss(output.squeeze(), target.squeeze())
        elif isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            target = target.to(torch.float)
            loss = self.loss(output.squeeze(), target.squeeze())
        else:
            loss = self.loss(output.squeeze(), target.squeeze())
        return loss

    def get_device(self) -> torch.device:
        """
        Return the device on which the classifier is actually loaded

        :return: device in use
        """
        return [*self.model.parameters()][0].device

    @abstractmethod
    def prune(self):
        """
        prune allows to delete the least important weights of the network such that the model is interpretable

        n: number of input features to retain
        """
        pass

    def fit(self, train_set: Dataset, val_set: Dataset, batch_size: int = 1024, epochs: int = 20, num_workers: int = 0,
            l_r: float = 0.01, lr_scheduler: bool = False, metric: Metric = TopkAccuracy(), early_stopping: bool = True,
            device: torch.device = torch.device("cpu"), verbose: bool = True, save: bool = True) -> pd.DataFrame:
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
        :param early_stopping: whether to perform early stopping (returning best model on val set) or not
        :param device: device on which to perform the training
        :param verbose: whether to output or not epoch metrics
        :param save: whether to save the model or not
        :return: pandas dataframe collecting the metrics from each epoch
        """

        # Setting device
        self.to(device), self.train()

        # Setting loss function and optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=l_r)
        scheduler = ReduceLROnPlateau(optimizer, verbose=verbose, mode='max', patience=epochs//10,
                                      factor=0.33, min_lr=1e-2 * l_r) if lr_scheduler else None

        # Training epochs
        best_acc, best_epoch = 0.0, -1
        train_accs, val_accs, tot_losses = [], [], []
        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=True,
                                                   num_workers=num_workers,
                                                   prefetch_factor=4 if num_workers != 0 else 2)

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            tot_losses_i = []
            train_outputs, train_labels = [], []
            for i, data in enumerate(train_loader):
                # Load batch (dataset, labels) on the correct device
                batch_data, batch_labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                # Network outputs on the current batch of dataset
                logit_outputs = self.forward(batch_data, logits=True)
                batch_outputs = self.activation(logit_outputs)

                # Compute losses and update gradients
                tot_loss = self.get_loss(logit_outputs, batch_labels, epoch, epochs)
                tot_loss.backward()
                optimizer.step()

                # Data moved to cpu again
                batch_outputs, batch_labels = batch_outputs.detach().cpu(), batch_labels.detach().cpu()
                tot_loss = tot_loss.detach().cpu()
                train_outputs.append(batch_outputs), train_labels.append(batch_labels), tot_losses_i.append(tot_loss)

            train_outputs, train_labels = torch.cat(train_outputs), torch.cat(train_labels)
            tot_losses_i = torch.stack(tot_losses_i).mean().item()

            # Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
            train_acc = self.evaluate(train_set, batch_size, metric, num_workers, device, train_outputs, train_labels)
            val_acc = self.evaluate(val_set, batch_size, metric, num_workers, device)

            # Save epoch results
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            tot_losses.append(tot_losses_i)

            # Step learning rate scheduler
            if lr_scheduler:
                scheduler.step(train_acc)

            # Save best model if early_stopping is True
            if (val_acc > best_acc and epoch >= epochs // 2 or epochs <= 2) and early_stopping:
                best_acc = val_acc
                best_epoch = epoch + 1
                self.save()

            # Prune network
            if (epoch + 1) == epochs // 2 and self.need_pruning:
                self.prune()
                self.need_pruning = False
                optimizer = torch.optim.AdamW(self.parameters(), lr=l_r)
                scheduler = ReduceLROnPlateau(optimizer, verbose=verbose, mode='max',
                                              factor=0.33, min_lr=1e-3 * l_r) if lr_scheduler else None

            if verbose:
                print(f"Epoch: {epoch + 1}/{epochs}, Loss: {tot_losses[-1]:.3f}, "
                      f"Tr_acc: {train_acc:.1f}, Val_acc: {val_acc:.1f}, best_e: {best_epoch}")

        # Best model is loaded and saved again with buffer "trained" set to true
        if early_stopping:
            self.load(device, set_trained=True)
            if not save:
                os.remove(self.name)
        elif save:
            self.save()

        # Performance dictionary
        performance_dict = {
            "Tot losses": tot_losses,
            "Train accs": train_accs,
            "Val accs": val_accs,
            "Best epoch": best_epoch,
        }
        performance_df = pd.DataFrame(performance_dict)
        return performance_df

    def evaluate(self, dataset: Dataset, batch_size: int = 1024,
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
                outputs, labels = self.predict(dataset, batch_size, num_workers, device)
            metric_val = metric(outputs.cpu(), labels.cpu())
        self.train()
        return metric_val

    def predict(self, dataset, batch_size: int = 1024, num_workers: int = 4,
                device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict function to compute the prediction of the model on a certain dataset

        :param dataset: dataset on which to test
        :param batch_size: number of training data for each step of the training
        :param num_workers: number of process to employ when loading data
        :param device: device on which to perform the training
        :return: a tuple containing the outputs computed on the dataset and the labels
        """
        self.to(device)
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
            incompat_keys = self.load_state_dict(torch.load(name, map_location=torch.device(device)), strict=False)
        except (FileNotFoundError, RuntimeError):
            raise ClassifierNotTrainedError()
        if set_trained:
            self.save(set_trained=True)
        if not self.trained:
            self._reinit()
            raise ClassifierNotTrainedError()
        if len(incompat_keys.missing_keys) > 0 or len(incompat_keys.unexpected_keys) > 0:
            if self.need_pruning:
                self.prune()
                incompat_keys = self.load_state_dict(torch.load(name, map_location=torch.device(device)), strict=False)
                if len(incompat_keys.missing_keys) > 0 or len(incompat_keys.unexpected_keys) > 0:
                    raise IncompatibleClassifierError(incompat_keys.missing_keys, incompat_keys.unexpected_keys)
            else:
                raise IncompatibleClassifierError(incompat_keys.missing_keys, incompat_keys.unexpected_keys)

    def _set_trained(self) -> None:
        """
        Internal function used to set the buffer "trained" to true
        """
        self.trained = torch.tensor(True)

    def _reinit(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


if __name__ == "__main__":
    pass
