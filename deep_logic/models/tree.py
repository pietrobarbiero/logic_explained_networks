import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset

from .base import BaseClassifier, ClassifierNotTrainedError, BaseXModel
from ..utils.base import tree_to_formula, NotAvailableError
from ..utils.metrics import Metric, Accuracy


class XDecisionTreeClassifier(BaseClassifier, BaseXModel):
    """
        Decision Tree class module. It does provides for explanations.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
        :param max_depth: int
            maximum depth for the classifier. The deeper is the tree, the more complex are the explanations provided.
     """

    def __init__(self, n_classes: int, n_features: int, max_depth: int = None,
                 device: torch.device = torch.device('cpu'), name: str = "tree.pth"):

        super().__init__(name=name, device=device)
        assert device == torch.device('cpu'), "Only cpu training is provided with decision tree models."

        self.n_classes = n_classes
        self.n_features = n_features

        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def forward(self, x, **kwargs) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        x = x.detach().cpu().numpy()
        output = self.model.predict_proba(x)
        return output

    def get_loss(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        """
        Loss is not used in the decision tree as it is not a gradient based algorithm. Therefore, if this function
        is called an error is thrown.
        :param output: output tensor from the forward function
        :param target: label tensor
        :param kwargs:
        :raise: NotAvailableError
        """
        raise NotAvailableError()

    def get_device(self) -> torch.device:
        """
        Return the device on which the classifier is actually loaded. For DecisionTree is always cpu

        :return: device in use
        """
        return torch.device("cpu")

    def fit(self, train_set: Dataset, val_set: Dataset, metric: Metric = Accuracy(),
            verbose: bool = True, save=True, **kwargs) -> pd.DataFrame:
        """
        fit function that execute many of the common operation generally performed by many method during training.
        Adam optimizer is always employed

        :param train_set: training set on which to train
        :param val_set: validation set used for early stopping
        :param metric: metric to evaluate the predictions of the network
        :param verbose: whether to output or not epoch metrics
        :param save: whether to save the model or not
        :return: pandas dataframe collecting the metrics from each epoch
        """

        # Loading dataset
        train_loader = torch.utils.data.DataLoader(train_set, 1024)
        train_data, train_labels = [], []
        for data in train_loader:
            train_data.append(data[0]), train_labels.append(data[1])
        train_data, train_labels = torch.cat(train_data).numpy(), torch.cat(train_labels).numpy()

        # Fitting decision tree
        if len(train_labels.squeeze().shape) > 1:
            train_labels = np.argmax(train_labels, axis=1)
        self.model = self.model.fit(X=train_data, y=train_labels)

        # Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
        train_acc = self.evaluate(train_set, metric=metric)
        val_acc = self.evaluate(val_set, metric=metric)

        if verbose:
            print(f"Train_acc: {train_acc:.1f}, Val_acc: {val_acc:.1f}")

        if save:
            self.save()

        # Performance dictionary
        performance_dict = {
            "tot_loss": [0],
            "train_accs": [train_acc],
            "val_accs": [val_acc],
            "best_epoch": [0],
        }
        performance_df = pd.DataFrame(performance_dict)
        return performance_df

    def predict(self, dataset, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict function to compute the prediction of the decision tree on a certain dataset

        :param dataset: dataset on which to test
        :return: a tuple containing the outputs computed on the dataset and the labels
        """
        outputs, labels = [], []
        loader = torch.utils.data.DataLoader(dataset, 1024)
        for data in loader:
            batch_data = data[0]
            batch_output = self.forward(batch_data)
            outputs.append(batch_output)
            labels.append(data[1].numpy())
        labels = np.concatenate(labels)
        outputs = np.vstack(outputs)
        return torch.FloatTensor(outputs), torch.FloatTensor(labels)

    def save(self, name=None, **kwargs) -> None:
        """
        Save model on a file named with the name of the model if parameter name is not set.

        :param name: Save the model with a name different from the one assigned in the __init__
        """
        from joblib import dump
        if name is None:
            name = self.name
        dump(self.model, name)

    def load(self, device=torch.device("cpu"), name=None, **kwargs) -> None:
        from joblib import load
        """
        Load decision tree model.

        :param name: Load a model with a name different from the one assigned in the __init__
        """
        if name is None:
            name = self.name
        try:
            self.model = load(name)
        except FileNotFoundError:
            raise ClassifierNotTrainedError() from None

    def prune(self):
        raise NotAvailableError()

    def get_local_explanation(self, **kwargs):
        raise NotAvailableError()

    def get_global_explanation(self, class_to_explain: int, concept_names: list = None, *args,
                               return_time: bool = False, **kwargs):
        if concept_names is None:
            concept_names = [f"f_{i}" for i in range(self.n_features)]
        start_time = time.time()
        formula = tree_to_formula(self.model, concept_names, class_to_explain)
        if return_time:
            return formula, time.time() - start_time
        return formula


if __name__ == "__main__":
    pass
