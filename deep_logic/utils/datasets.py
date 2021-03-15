import os
import json
from abc import ABC, abstractmethod
from collections import Callable

import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from data import CUB200


class ConceptDataset(ImageFolder, ABC):
    """
    Simple abstract dataset to use for working with Concepts that extend ImageFolder dataset from torchvision.

    :param root: path to the main folder of the dataset
    :param transform: transforms to use on the images
    :param dataset_name: name of the dataset (useful for saving purpose)
    :param predictions: whether to use predictions of a network as attributes or the attributes labels
    :param denoised: whether to use original attributes or denoised attributes as done in literature.

    """
    def __init__(self, root: str, transform: Callable = None, dataset_name: str = "CUB200",
                 predictions: bool = False, denoised: bool = False):
        super().__init__(root, transform)
        from .data import clean_names
        assert not predictions or not denoised, "Using predictions as attributes " \
                                                "does not allow using denoised attributes"

        if predictions:
            self.attributes = np.load(os.path.join(root, dataset_name + "_predictions.npy"))
        elif denoised:
            self.attributes = np.load(os.path.join(root, "very_denoised_attributes.npy"))
            attribute_to_filter = np.sum(self.attributes, axis=0) > 0
            self.attributes = self.attributes[:, attribute_to_filter]
        else:
            self.attributes = np.load(os.path.join(root, "attributes.npy"))

        with open(os.path.join(root, "attributes_names.txt"), "r") as f:
            self.attribute_names = json.load(f)
            self.attribute_names = np.asarray(clean_names(self.attribute_names))
        if (predictions or denoised) and dataset_name == CUB200:
            attributes = np.load(os.path.join(root, "very_denoised_attributes.npy"))
            attribute_to_filter = np.sum(attributes, axis=0) > 0
            self.attribute_names = self.attribute_names[attribute_to_filter]

        self.attributes = self.attributes.astype(np.float32)
        self.n_attributes = self.attributes.shape[1]
        self.n_classes = len(self.classes)
        self.dataset_name = dataset_name
        self.targets = np.asarray(self.targets)

    @abstractmethod
    def __getitem__(self, idx):
        pass


class ConceptToTaskDataset(ConceptDataset):
    """
    Extension of ConceptDataset to use for working with Concepts as inputs to predict final class.

    """
    def __init__(self, root: str, dataset_name: str = "CUB200", predictions: bool = True, denoised: bool = False):
        super().__init__(root, dataset_name=dataset_name, predictions=predictions, denoised=denoised)

    def __getitem__(self, idx):
        """
        Instead of returning image, class it returns attributes, class

        """
        target = self.targets[idx]
        attribute = self.attributes[idx]
        return attribute, target


class ImageToConceptDataset(ConceptDataset):
    """
    Extension of ConceptDataset to use for working with Concepts as labels instead of final classes.

    """
    def __init__(self, root: str, transform: transforms, dataset_name: str = "CUB200", denoised: bool = False):
        super().__init__(root, transform, dataset_name=dataset_name, predictions=False, denoised=denoised)

    def __getitem__(self, idx):
        """
        Instead of returning image, class it returns image, attributes

        """
        sample, _ = ImageFolder.__getitem__(self, idx)
        attribute = self.attributes[idx]
        return sample, attribute
