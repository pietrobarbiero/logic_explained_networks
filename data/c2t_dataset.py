import json
import os
import numpy as np
from torchvision.datasets import ImageFolder


class ConceptToTaskDataset(ImageFolder):
    """
    Simple Dataset to use for Concept extraction that extend ImageFolder dataset from torchvision.
    Instead of returning the single class the image belongs to, it returns the array of visible attributes on the image.

    :param root: path to the main folder of the dataset
    """
    def __init__(self, root: str, dataset_name: str = "CUB200", denoised: bool = False):
        super().__init__(root)
        if denoised:
            self.attributes = np.load(os.path.join(root, "very_denoised_attributes.npy"))
        else:
            self.attributes = np.load(os.path.join(root, "attributes.npy"))
        with open(os.path.join(root, "attributes_names.txt"), "r") as f:
            self.attribute_names = json.load(f)
        self.attributes = self.attributes.astype(np.float32)
        self.n_attributes = self.attributes.shape[1]
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        _, target = super().__getitem__(idx)
        attribute = self.attributes[idx]
        return attribute, target
