import collections
import time
import warnings
from multiprocessing import Pool
from typing import Tuple, List

import numpy
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sympy import to_dnf, simplify_logic
from torch.utils.data import Dataset
from tqdm import tqdm

from lens.logic import replace_names, test_explanation
from lens.models.ext_models.anchor.anchor_tabular import AnchorTabularExplainer
from lens.utils.datasets import StructuredDataset
from .base import BaseClassifier, ClassifierNotTrainedError, BaseXModel
from ..utils.base import NotAvailableError, to_categorical
from ..utils.metrics import Metric, Accuracy, F1Score


class XAnchorClassifier(BaseClassifier, BaseXModel):
    """
        Anchor-explained Black Box classifier module. It does provides for explanations.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
     """

    def __init__(self, n_classes: int, n_features: int, train_dataset: StructuredDataset,
                 device: torch.device = torch.device('cpu'),
                 name: str = "anchor.pth"):

        super().__init__(name=name, device=device)
        assert device == torch.device('cpu'), "Only cpu training is provided with decision tree models."

        self.n_classes = n_classes
        self.n_features = n_features
        self.dataset = train_dataset

        self.model = RandomForestClassifier(n_estimators=50, n_jobs=5)
        self.explainer = AnchorTabularExplainer(
            train_dataset.class_names,
            train_dataset.feature_names,
            train_dataset.x.numpy(),
        )

        if n_classes == 1:
            n_classes = 2
        self.explanations = ["" for _ in range(n_classes)]

    def forward(self, x, return_np=False, single_output=False, **kwargs) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param single_output:
        :param x: input tensor
        :param return_np:
        :return: output classification
        """
        x = self._check_input(x)
        output = self.model.predict_proba(x)
        if output.shape == 1:
            output = np.expand_dims(output, 0)
        if single_output:
            output = np.argmax(output, axis=1)
        if return_np:
            return output
        return torch.as_tensor(output)

    @staticmethod
    def _check_input(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            assert isinstance(x, np.ndarray), "Only numpy array or torch.Tensor can be passed"
        return x

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
        train_loader = torch.utils.data.DataLoader(train_set, len(train_set))
        train_data, train_labels = train_loader.__iter__().next()

        # Checking labels: if multi class (but not multilabel) reduce to a single array
        if len(train_labels.squeeze().shape) > 1 and train_labels.sum() == train_labels.shape[0]:
            train_labels = np.argmax(train_labels, axis=1)

        # Fitting decision tree
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
        train_loader = torch.utils.data.DataLoader(dataset, len(dataset))
        data = train_loader.__iter__().next()
        output = self.model.predict(data[0])
        # multilabel output
        if isinstance(output, list):
            output = np.asarray(output).reshape((data[1].shape + (-1,))).argmax(axis=2)
        labels = data[1].numpy()
        return torch.FloatTensor(output), torch.FloatTensor(labels)

    def save(self, device=torch.device("cpu"), name=None, **kwargs) -> None:
        """
        Save model on a file named with the name of the model if parameter name is not set.

        :param device:
        :param name: Save the model with a name different from the one assigned in the __init__
        """
        from joblib import dump
        if name is None:
            name = self.name
        checkpoint = {
            "model": self.model,
            "explanations": self.explanations,
            "time": self.time
        }
        dump(checkpoint, name)

    def load(self, device=torch.device("cpu"), name=None, **kwargs) -> None:
        """
        Load decision tree model.

        :param device:
        :param name: Load a model with a name different from the one assigned in the __init__
        """
        from joblib import load
        if name is None:
            name = self.name
        try:
            checkpoint = load(name)
            if 'model' in checkpoint:
                self.model = checkpoint['model']
                self.explanations = checkpoint['explanations']
                self.time = checkpoint['time']
            else:
                self.model = checkpoint
                warnings.warn("Loaded model does not have time or explanations. "
                              "They need to be recalculated but time will only consider rule extraction time.")
        except FileNotFoundError:
            raise ClassifierNotTrainedError() from None

    def prune(self):
        raise NotAvailableError()

    def get_local_explanation(self, x_test, idx=None, **kwargs):
        if idx is not None:
            print(idx)
        x_test = self._check_input(x_test)
        exp = self.explainer.explain_instance(x_test, lambda x: self(x,
                                                                     return_np=True,
                                                                     single_output=True),
                                              threshold=0.95)
        logic_exp = "".join([name + " & " for name in exp.names()])[:-3]
        return logic_exp

    def get_global_explanation(self, target_class: int, val_set: StructuredDataset,
                               *args, return_time: bool = False, **kwargs):
        start_time = time.time()
        if self.explanations[target_class] != "":
            explanation = self.explanations[target_class]
        else:
            explanation, _ = self._combine_local_explanations(target_class,
                                                              x_val=val_set.x,
                                                              y_val=val_set.y)
            self.explanations[target_class] = explanation

        if return_time:
            return explanation, time.time() - start_time
        return explanation

    def _combine_local_explanations(self, target_class: int, topk_explanations=None,
                                    return_accuracy: bool = False, metric: Metric = F1Score(),
                                    x_val: torch.tensor = None, y_val=None):
        """
        Generate a global explanation combining local explanations.

        :param metric:
        :param target_class: class ID
        :param topk_explanations: number of most common local explanations to combine in a global explanation (it controls
                the complexity of the global explanation)
        :param return_accuracy: whether to return also the accuracy of the explanations or not
        :param x_val:
        :param y_val:
        :return: Global explanation, predictions, and ranking of local explanations
        """
        assert topk_explanations is not None or (x_val is not None and y_val is not None), \
            "validation data need to be passed when the number of top explanations to retain is not specified"
        # rand_idx = np.random.randint(0, self.dataset.x.shape[0], )
        x, y = self.dataset.x, self.dataset.y
        concept_names: List = self.dataset.feature_names

        # if x_val is None or y_val is None:
        #     x_val, y_val = x, y
        y = to_categorical(y)
        y_val = to_categorical(y_val)

        assert (y == target_class).any(), "Cannot get explanation if target class is not amongst target labels"
        x_target = x[y == target_class]
        y_target = y[y == target_class]

        # get model's predictions
        preds = self(x_target)
        preds = to_categorical(preds)

        # identify samples correctly classified of the target class
        correct_mask = y_target.eq(preds)
        x_target_correct = x_target[correct_mask]
        y_target_correct = y_target[correct_mask]

        # collapse samples having the same class label different from the target class
        _, idx = np.unique((x[y != target_class] > 0.5).cpu().detach().numpy(), axis=0,
                           return_index=True)
        if topk_explanations is None:
            x_reduced_opposite = x[y != target_class][idx]
            y_reduced_opposite = y[y != target_class][idx]
        else:
            x_reduced_opposite = x[y != target_class]
            y_reduced_opposite = y[y != target_class]
        preds_opposite = self(x_reduced_opposite)
        if len(preds_opposite.squeeze(-1).shape) > 1:
            preds_opposite = torch.argmax(preds_opposite, dim=1)
        else:
            preds_opposite = (preds_opposite > 0.5).squeeze()

        # identify samples correctly classified of the opposite class
        correct_mask = y_reduced_opposite.eq(preds_opposite)
        x_reduced_opposite_correct = x_reduced_opposite[correct_mask]
        y_reduced_opposite_correct = y_reduced_opposite[correct_mask]

        # select the subset of samples belonging to the target class
        x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
        y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

        # generate local explanation only for samples where:
        # 1) the model's prediction is correct and
        # 2) the class label corresponds to the target class
        # with Pool() as pool:
        #     local_explanations = pool.map(self.get_local_explanation,
        #                                   zip(x_target_correct, range(x_target_correct.shape[0])))
        local_explanations = []
        for sample_id, (xi, yi) in enumerate(tqdm(zip(x_target_correct, y_target_correct),
                                                  desc="Extracting anchors explanation",
                                                  total=x_target_correct.shape[0])):
            local_explanation_raw = self.get_local_explanation(xi)
            local_explanations.append(local_explanation_raw)

        if len(local_explanations) == 0:
            if not return_accuracy:
                return '', np.array, collections.Counter()
            else:
                return '', np.array, collections.Counter(), 0.

        # get most frequent local explanations
        counter = collections.Counter(local_explanations)
        if topk_explanations is None or len(counter) < topk_explanations:
            topk_explanations = len(counter)

        most_common_explanations = []
        best_accuracy = 0
        for i, (explanation, _) in enumerate(counter.most_common(topk_explanations)):
            if explanation in ['', 'False', 'True']:
                continue
            most_common_explanations.append(explanation)
            global_explanation = ' | '.join(most_common_explanations)
            if x_val is not None and y_val is not None:
                accuracy, predictions = test_explanation(global_explanation, target_class,
                                                         x_val, y_val, metric=metric,
                                                         concept_names=concept_names,
                                                         inequalities=True)
                if accuracy <= best_accuracy:
                    most_common_explanations.remove(explanation)
                else:
                    best_accuracy = accuracy

        # the global explanation is the disjunction of local explanations
        global_explanation = ' | '.join(most_common_explanations)

        # predictions based on FOL formula
        accuracy, predictions = test_explanation(global_explanation, target_class,
                                                 x_validation, y_validation,
                                                 metric=metric, concept_names=concept_names,
                                                 inequalities=True)

        # replace concept names
        if concept_names is not None:
            global_explanation = replace_names(global_explanation, concept_names)

        if not return_accuracy:
            return global_explanation, predictions
        else:
            return global_explanation, predictions, accuracy


if __name__ == "__main__":
    pass
