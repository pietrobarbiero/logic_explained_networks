import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from lens.utils.datasets import MyDataset
from lens.utils.data import ConceptDataset
from lens.models.base import BaseClassifier, BaseXModel
from lens.models.ext_models import deep_red
from lens.utils.base import NotAvailableError
from lens.utils.metrics import Metric, Accuracy


class XDeepRedClassifier(BaseClassifier, BaseXModel):
    """
        Deep Red class module. It does provides for explanations.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
        :param function:
            activation function, 'tanh' or 'sigmoid'
        :param softmax:
            whether to use a softmax cross entropy or a mean square error loss

     """

    def __init__(self, n_classes: int, n_features: int, function: str = "tanh", softmax: bool = True,
                 name: str = "deep_red.pth", device: torch.device = torch.device('cpu')):

        assert device == torch.device('cpu'), "Only cpu training is provided with Deep Red models."
        assert function == "tanh" or function == "sigmoid", "Only sigmoid or tanh activation functions " \
                                                            "are provided by DeepRed"
        self._origin_folder = os.path.abspath(os.curdir)
        self._result_folder = os.path.dirname(name)
        os.chdir(self._result_folder)
        name = os.path.basename(name)
        super().__init__(name=name, device=device)

        self.n_classes = n_classes
        self.n_features = n_features
        self.function = function
        self.softmax = softmax

        # Network structures from DEEP RED paper
        self.hidden_nodes = [
            min(30, 2 * n_features),
            (n_features + n_classes) // 2,
            n_classes
        ]
        self.data = None
        self._prepared_data = False
        self.dataset_name = None
        self.split_name = None
        self._prepare_folders()

        if n_classes == 1:
            n_classes = 2
        self.explanations = ["" for _ in range(n_classes)]

    @staticmethod
    def _prepare_folders():
        model_folder = "models"
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
            print("Created model folder")
        data_folder = "data"
        if not os.path.isdir(data_folder):
            os.mkdir(data_folder)
            print("Created data folder")
        obj_folder = "obj"
        if not os.path.isdir(obj_folder):
            os.mkdir(obj_folder)
            print("Created obj folder")
        indexes_folder = "indexes"
        if not os.path.isdir(indexes_folder):
            os.mkdir(indexes_folder)
            print("Created index folder")

    def prepare_data(self, dataset: MyDataset, dataset_name: str, seed: int, train_idx=None, test_idx=None,
                     train_sample_rate=1.):
        """
        :param dataset:
        :param dataset_name:
        :param seed:
        :param train_idx:
        :param test_idx:
        :param train_sample_rate:
        """

        dataset.save_as_csv("data")
        self.dataset_name = dataset_name
        self.split_name = f"{seed}_{train_sample_rate}"

        if train_idx is None or test_idx is None:
            deep_red.main.set_split(self.dataset_name, self.split_name, (100 - int(1/train_sample_rate)*100))
            train_idx, test_idx = deep_red.lr.load_indexes(self.dataset_name, self.split_name)
        else:
            data, labels = next(torch.utils.data.DataLoader(dataset, len(dataset)).__iter__())
            train_data, train_labels = data[train_idx], labels[train_idx]
            train_idx = self._random_sample_data(train_sample_rate, train_data, train_labels, return_idx=True)
            deep_red.main.set_split_manually(self.dataset_name, self.split_name,
                                             train_indexes=train_idx, test_indexes=test_idx)
        self.data = deep_red.obj_data_set.DataSet(self.dataset_name, self.hidden_nodes)
        self.data.set_split(train_idx, [], test_idx)

        self._prepared_data = True
        print("Train idx len:", len(train_idx))
        print("Test idx len:", len(test_idx))

    def forward(self, x, **kwargs) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        raise NotAvailableError()

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

    def fit(self, epochs: int = 1000, metric: Metric = Accuracy(), verbose: bool = False, save=True,
            **kwargs) -> pd.DataFrame:
        """
        fit function that execute many of the common operation generally performed by many method during training.
        Adam optimizer is always employed

        :param epochs: number of epochs to train the model
        :param metric: metric to evaluate the predictions of the network
        :param verbose: whether to output or not epoch metrics
        :param save: whether to save the model or not
        :return: pandas dataframe collecting the metrics from each epoch
        """
        assert self._prepared_data, "In Deep Red, data need to be prepared before training"

        old_stdout = sys.stdout
        if not verbose:
            sys.stdout = None
        model_name = os.path.basename(self.name)
        deep_red.main.prepare_network(self.dataset_name, self.split_name, model_name, self.hidden_nodes,
                                      softmax=self.softmax, init_iterations=epochs)
        sys.stdout = old_stdout

        train_acc = self.evaluate()
        if verbose:
            print(f"Train_acc: {train_acc:.1f}")

        if save:
            self.save(set_trained=True)

        # Performance dictionary
        performance_dict = {
            "tot_loss": [0],
            "train_accs": [train_acc],
            "val_accs": [train_acc],
            "best_epoch": [0],
        }
        performance_df = pd.DataFrame(performance_dict)
        return performance_df

    def evaluate(self, train: bool = True, outputs: torch.tensor = None, labels: torch.tensor = None,
                 metric: Metric = Accuracy(), *args, **kwargs) -> float:
        """
        Evaluate function to test the performance of the model on a certain dataset without training
        :param outputs:
        :param labels:
        :param train:
        :param metric:
        :return: metric evaluated on the dataset
        """
        self.eval()
        with torch.no_grad():
            if outputs is None or labels is None:
                outputs, labels = self.predict(train)
            metric_val = metric(outputs.cpu(), labels.cpu())
        self.train()
        return metric_val

    def predict(self, train: bool = True, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict function to compute the prediction of the decision tree on a certain dataset

        :return: a tuple containing the outputs computed on the dataset and the labels
        """
        assert self._prepared_data, "In Deep Red, data need to be prepared before predicting"
        old_stdout = sys.stdout
        sys.stdout = None
        model_name = os.path.basename(self.name)
        act_train, _, act_test, _, _, _ = deep_red.dnnt.execute_network(self.data, model_name, self.hidden_nodes,
                                                                        function=self.function, softmax=self.softmax)
        sys.stdout = old_stdout
        if train:
            output = act_train[-1]
            _, labels = self.data.get_train_x_y()
        else:
            output = act_test[-1]
            _, labels = self.data.get_test_x_y()
        labels = np.argmax(labels, axis=1)

        return torch.as_tensor(output), torch.as_tensor(labels)

    def finish(self):
        os.chdir(self._origin_folder)

    def prune(self):
        raise NotAvailableError()

    def get_local_explanation(self, **kwargs):
        raise NotAvailableError()

    def get_global_explanation(self, target_class: int, concept_names: list = None, *args,
                               return_time: bool = False, verbose=False, **kwargs):

        print(f"Extracting explanation for class {target_class}")
        assert self.trained, "Model need to be trained before extracting explanations"
        t = time.time()
        if self.explanations[target_class] != "":
            explanation = self.explanations[target_class]
        else:
            old_stdout = sys.stdout
            if not verbose:
                sys.stdout = None
            model_name = os.path.basename(self.name)
            exp_accuracy, fidelity, complexity, bio = deep_red.main.extract_model(self.dataset_name, self.split_name,
                                                                                  model_name, self.hidden_nodes,
                                                                                  target_class)
            sys.stdout = old_stdout
            explanation = deep_red.main.convert_rule(bio, concept_names, self.n_features)
            self.explanations[target_class] = explanation

        print(f"Finished extracting explanation for class {target_class}")
        if return_time:
            return explanation, time.time() - t
        return explanation


if __name__ == "__main__":
    pass
