import os
import json
from abc import ABC, abstractmethod
from collections import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from data import CUB200
from deep_logic.utils.base import NotAvailableError


class ConceptDataset(ImageFolder, ABC):
    """
    Simple abstract dataset to use for working with Concepts that extend ImageFolder dataset from torchvision.

    :param root: path to the main folder of the dataset
    :param transform: transforms to use on the images
    :param dataset_name: name of the dataset (useful for saving purpose)
    :param predictions: whether to use predictions of a network as attributes or the attributes labels
    :param multi_label: for multi-label dataset (classes + attributes)
    """

    def __init__(self, root: str, transform: Callable = None, dataset_name: str = CUB200,
                 predictions: bool = False, multi_label: bool = False, binary=False):
        super().__init__(root, transform)
        from .data import clean_names

        if predictions:
            if multi_label:
                self.attributes = np.load(os.path.join(root, dataset_name + "_multi_label_predictions.npy"))
            else:
                self.attributes = np.load(os.path.join(root, dataset_name + "_predictions.npy"))
        else:
            self.attributes = np.load(os.path.join(root, "attributes.npy"))
            if multi_label:
                multi_labels_targets = LabelBinarizer().fit_transform(self.targets)
                if np.unique(np.asarray(self.targets)).shape[0] == 2:
                    multi_labels_targets = np.hstack((1 - multi_labels_targets, multi_labels_targets))
                self.attributes = np.concatenate((multi_labels_targets, self.attributes), axis=1)

        self.targets = np.asarray(self.targets)
        if binary:
            if len(self.targets.squeeze().shape) == 1:
                targets = LabelBinarizer().fit_transform(self.targets)
                if len(targets.squeeze().shape) == 1:
                    targets = np.hstack([1 - targets, targets])
                self.targets = targets

        with open(os.path.join(root, "attributes_names.txt"), "r") as f:
            self.attribute_names : list = json.load(f)
            if multi_label:
                self.attribute_names = self.classes + self.attribute_names
            self.attribute_names = np.asarray(clean_names(self.attribute_names))

        self.attributes = self.attributes.astype(np.float32)
        self.n_attributes = self.attributes.shape[1]
        self.n_classes = len(self.classes)
        self.dataset_name = dataset_name
        self.targets = np.asarray(self.targets)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def save_as_csv(self, folder=None):
        pass


class ConceptOnlyDataset(ConceptDataset):
    """
     Extension of ConceptDataset to use for Concept only analysis (e.g. clustering).

    """

    def __init__(self, root: str, predictions=True, **kwargs):
        super().__init__(root, predictions=predictions, multi_label=True, **kwargs)

    def __getitem__(self, idx):
        attribute = self.attributes[idx]
        target = self.targets[idx]
        return attribute, target

    def save_as_csv(self, folder=None):
        raise NotImplementedError()


class ConceptToTaskDataset(ConceptDataset):
    """
    Extension of ConceptDataset to use for working with Concepts as inputs to predict final class.

    """

    def __init__(self, root: str, predictions: bool = True, **kwargs):
        super().__init__(root, predictions=predictions, multi_label=False, **kwargs)

    def __getitem__(self, idx):
        """
        Instead of returning image, class it returns attributes, class

        """
        target = self.targets[idx]
        attribute = self.attributes[idx]
        return attribute, target

    def save_as_csv(self, folder=None):
        if folder is None:
            folder = self.root
        csv_name = os.path.join(folder, self.dataset_name + ".csv")
        if not os.path.isfile(csv_name):
            attributes = {
                concept_name: self.attributes[:, i] for i, concept_name in enumerate(self.attribute_names)
            }
            attributes.update({"output_class": self.targets})
            df = pd.DataFrame(attributes)
            df.to_csv(csv_name, index=False, header=False)


class ImageToConceptDataset(ConceptDataset):
    """
    Extension of ConceptDataset to use for working with Concepts as labels instead of final classes.

    """

    def __init__(self, root: str, transform: transforms, **kwargs):
        super().__init__(root, transform, predictions=False, multi_label=False, **kwargs)

    def __getitem__(self, idx):
        """
        Instead of returning image, class it returns image, attributes

        """
        sample, _ = ImageFolder.__getitem__(self, idx)
        attribute = self.attributes[idx]
        return sample, attribute

    def save_as_csv(self, folder=None):
        raise NotAvailableError("Cannot save dataset as csv when working with images")


class ImageToConceptAndTaskDataset(ConceptDataset):
    """
     Extension of ConceptDataset to use for multi-label classification (Task + Concepts).

    """
    def __init__(self, root: str, transform: transforms, **kwargs):
        super().__init__(root, transform, predictions=False, multi_label=True, **kwargs)

    def __getitem__(self, idx):
        sample, _ = ImageFolder.__getitem__(self, idx)
        attribute = self.attributes[idx]
        return sample, attribute

    def save_as_csv(self, folder=None):
        raise NotAvailableError("Cannot save dataset as csv when working with images")
