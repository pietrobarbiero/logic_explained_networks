import time
from concurrent.futures.process import ProcessPoolExecutor
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .base import BaseClassifier, ClassifierNotTrainedError, BaseXModel
from .ext_models.brl.RuleListClassifier import RuleListClassifier
from ..utils.base import NotAvailableError, brl_extracting_formula
from ..utils.metrics import Metric, Accuracy


class XBRLClassifier(BaseClassifier, BaseXModel):
    """
        BR class module. It does provides for explanations.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
        :param discretize: bool
            whether to discretize data or not
        :param feature_names: list
            name of the features, if not given substituted with f1, f2, ... fd
        :param class_names: list
            name of the classes, if not given substituted with class_1, class_2, ... class_n
    """

    def __init__(self, n_classes: int, n_features: int, discretize: bool = False, feature_names: list = None,
                 class_names: list = None, device: torch.device = torch.device('cpu'), name: str = "brl.pth", ):

        super().__init__(name=name, device=device)
        assert device == torch.device('cpu'), "Only cpu training is provided with BRL models."

        self.n_classes = n_classes
        self.n_features = n_features
        self.discretize = discretize

        self.model = []
        self.class_names = []
        for i in range(self.n_classes):
            class_name = class_names[i] if class_names is not None else f"class_{i}"
            model = RuleListClassifier(max_iter=10000, class1label=class_name, verbose=False)
            self.model.append(model)
            self.class_names.append(class_name)

        if feature_names is not None:
            self.features_names = feature_names
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
        if self.discretize:
            x = self._discretize(x)
        outputs = []
        pbar = tqdm(range(self.n_classes), desc="BRL predicting classes")
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.n_classes):
                args = {
                    "self": self.model[i],
                    "X": x,
                }
                futures.append(executor.submit(RuleListClassifier.predict_proba, **args))
            for i in range(self.n_classes):
                brl_outputs = futures[i].result()

                # BRL outputs both the negative prediction (output[0]) and the positive (output[1])
                output = brl_outputs[:, 1]
                outputs.append(torch.tensor(output))
                pbar.update()
        pbar.close()

        outputs = torch.stack(outputs, dim=1)

        return outputs

    def _discretize(self, train_data: np.ndarray) -> np.ndarray:
        train_data = [[self.features_names[i] if item > 0.5 else "~" + self.features_names[i]
                       for i, item in enumerate(array)]
                      for array in train_data]
        return np.asarray(train_data)

    def _binarize_labels(self, train_labels: torch.tensor):
        if len(train_labels.shape) == 1:
            train_labels = np.expand_dims(train_labels, axis=1)
        if len(np.unique(train_labels)) > 2:
            train_labels = LabelBinarizer().fit_transform(train_labels)
            print(f"Binarized labels. Labels {train_labels.shape}")
        elif len(np.unique(train_labels)) == 2 and self.n_classes == 2:
            train_labels = np.hstack((1 - train_labels, train_labels))
        else:
            print(f"Labels {train_labels.shape}")
        return train_labels

    def get_loss(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        """
        Loss is not used in BRL as it is not a gradient based algorithm. Therefore, if this function
        is called an error is thrown.
        :param output: output tensor from the forward function
        :param target: label tensor
        :param kwargs:
        :raise: NotAvailableError
        """
        raise NotAvailableError()

    def get_device(self) -> torch.device:
        """
        Return the device on which the classifier is actually loaded. For BRL is always cpu

        :return: device in use
        """
        return torch.device("cpu")

    def fit(self, train_set: Dataset, val_set: Dataset = None, metric: Metric = Accuracy(),
            verbose: bool = True, save=True, eval=True, **kwargs) -> pd.DataFrame:
        """
        fit function that execute many of the common operation generally performed by many method during training.
        Adam optimizer is always employed

        :param train_set: training set on which to train
        :param val_set: validation set used for early stopping
        :param metric: metric to evaluate the predictions of the network
        :param verbose: whether to output or not epoch metrics
        :param save: whether to save the model or not
        :param eval: whether to evaluate training and validation data (it may takes time)
        :return: pandas dataframe collecting the metrics from each epoch
        """

        # Loading dataset
        train_loader = torch.utils.data.DataLoader(train_set, 1024)
        train_data, train_labels = [], []
        for data in train_loader:
            train_data.append(data[0]), train_labels.append(data[1])
        train_data = torch.cat(train_data).numpy()
        train_labels = torch.cat(train_labels).numpy()
        train_labels = self._binarize_labels(train_labels)

        if self.discretize:
            print("Discretized features")
            train_data = self._discretize(train_data)
            features = self.features_names
        else:
            features = []

        # Fitting a BRL classifier for each class in parallel
        with ProcessPoolExecutor() as executor:
            futures = []
            pbar = tqdm(range(self.n_classes), desc="BRL training classes")
            for i in range(self.n_classes):
                self.model[i].verbose = verbose
                args = {
                    "self": self.model[i],
                    "X" : train_data,
                    "y" : train_labels[:, i],
                    "feature_labels" : self.features_names,
                    "undiscretized_features": features
                }
                # RuleListClassifier.fit(**args)
                futures.append(executor.submit(RuleListClassifier.fit, **args))
            for i, future in enumerate(futures):
                self.model[i] = future.result()
                pbar.update()
            pbar.close()

        # Compute accuracy, f1 and constraint_loss on the whole train, validation dataset
        if eval:
            train_acc = self.evaluate(train_set, metric=metric)
            if val_set is not None:
                val_acc = self.evaluate(val_set, metric=metric)
            else:
                val_acc = 0
        else:
            train_acc, val_acc = 0., 0.

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

    def predict(self, dataset: Dataset, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict function to compute the prediction of BRL on a certain dataset

        :param dataset: dataset on which to test
        :return: a tuple containing the outputs computed on the dataset and the labels
        """
        outputs, labels = [], []
        loader = torch.utils.data.DataLoader(dataset, 2**20)
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
        Load BRL model.

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
        start_time = time.time()

        # formula = str(self.model[class_to_explain])
        formula = brl_extracting_formula(self.model[class_to_explain])

        if concept_names is not None:
            for i, name in enumerate(concept_names):
                formula = formula.replace(f"ft{i}", name)

        if return_time:
            return formula, time.time() - start_time
        return formula


if __name__ == "__main__":
    pass
