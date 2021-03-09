import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset

from .base import BaseClassifier, ClassifierNotTrainedError, BaseXModel
from .ext_models.brl.RuleListClassifier import RuleListClassifier
from ..utils.base import tree_to_formula, NotAvailableError
from ..utils.metrics import Metric, Accuracy


class XBRLClassifier(BaseClassifier, BaseXModel):
    """
        Decision Tree class module. It does provides for explanations.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
     """

    def __init__(self, n_classes: int, n_features: int, max_depth: int = None,
                 device: torch.device = torch.device('cpu'), name: str = "brl.pth", features_names=None,
                 class_names=None):

        super().__init__(name=name, device=device)
        assert device == torch.device('cpu'), "Only cpu training is provided with decision tree models."

        self.n_classes = n_classes
        self.n_features = n_features

        self.model = []
        self.class_names = []
        for i in range(self.n_classes):
            class_name = class_names[i] if class_names is not None else f"class_{i}"
            model = RuleListClassifier(max_iter=10000, class1label=class_name, verbose=False)
            self.model.append(model)
            self.class_names.append(class_name)

        if features_names is not None:
            self.features_names = features_names
        else:
            self.features_names = [f"f{i}" for i in range(n_features)]

    def forward(self, x, **kwargs) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        x = x.detach().cpu().numpy()
        outputs = []
        for i in range(self.n_classes):
            output = self.model[i].predict_proba(x)
            output = np.argmax(output, axis=1)
            outputs.append(torch.tensor(output))
        outputs = torch.stack(outputs, dim=1)

        return outputs

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

    def fit(self, train_set: Dataset, val_set: Dataset, metric: Metric = Accuracy(), discretize: bool = True,
            verbose: bool = True, save=True, **kwargs) -> pd.DataFrame:
        """
        fit function that execute many of the common operation generally performed by many method during training.
        Adam optimizer is always employed

        :param train_set: training set on which to train
        :param val_set: validation set used for early stopping
        :param metric: metric to evaluate the predictions of the network
        :param discretize: whether to discretize data or not
        :param verbose: whether to output or not epoch metrics
        :param save: whether to save the model or not
        :return: pandas dataframe collecting the metrics from each epoch
        """

        # Loading dataset
        train_loader = torch.utils.data.DataLoader(train_set, 1024, num_workers=8)
        train_data, train_labels = [], []
        for data in train_loader:
            train_data.append(data[0]), train_labels.append(data[1])
        train_data = torch.cat(train_data).numpy()
        # train_data = np.concatenate((train_data, train_data, train_data))
        # train_data = train_data.astype(str)
        train_labels = torch.cat(train_labels).numpy()
        # train_labels = np.concatenate((train_labels, train_labels, train_labels))

        if len(train_labels.shape) == 1:
            train_labels = np.expand_dims(train_labels, axis=1)

        if discretize:
            features = []
        else:
            features = self.features_names

        # Fitting a BRL classifier for each class
        for i in range(self.n_classes):
            self.model[i].verbose = verbose
            self.model[i].fit(X=train_data, y=train_labels[:, i], feature_labels=self.features_names,
                              undiscretized_features=features)

        # Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
        train_acc = self.evaluate(train_set, metric)
        val_acc = self.evaluate(val_set, metric)

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

    def evaluate(self, dataset: Dataset, metric: Metric = Accuracy(), **kwargs) -> float:
        """
        Evaluate function to test without training the performance of the decision tree on a certain dataset

        :param dataset: dataset on which to test
        :param metric: metric to evaluate the predictions of the network
        :return: metric evaluated on the dataset
        """
        outputs, labels = self.predict(dataset)
        metric_val = metric(outputs, labels)
        return metric_val

    def predict(self, dataset, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def load(self, name=None, **kwargs) -> None:
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
        formula = str(self.model[class_to_explain])
        if concept_names is not None:
            for i, name in enumerate(concept_names):
                formula = formula.replace(f"ft{i}", name)
        return formula


if __name__ == "__main__":
    pass
