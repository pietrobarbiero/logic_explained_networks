import os
import json
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


class ImageToConceptDataset(ImageFolder):
    """
    Simple Dataset to use for Concept extraction that extend ImageFolder dataset from torchvision.
    Instead of returning the single class the image belongs to, it returns the array of visible attributes on the image.

    :param root: path to the main folder of the dataset
    :param transform: transform to apply to the image
    """
    def __init__(self, root: str, transform: transforms, dataset_name: str = "CUB200", denoised: bool = False):
        super().__init__(root, transform)
        self.dataset_name = dataset_name
        if denoised:
            self.attributes = np.load(os.path.join(root, "very_denoised_attributes.npy"))
        else:
            self.attributes = np.load(os.path.join(root, "attributes.npy"))
        self.attributes = self.attributes.astype(np.float32)
        with open(os.path.join(root, "attributes_names.txt"), "r") as f:
            self.attribute_names = np.asarray(json.load(f))
        if denoised:
            attribute_to_filter = np.sum(self.attributes, axis=0) > 0
            self.attributes = self.attributes[:, attribute_to_filter]
            self.attribute_names = self.attribute_names[attribute_to_filter]
        self.n_attributes = self.attributes.shape[1]

    def __getitem__(self, idx):
        sample, _ = super().__getitem__(idx)
        attribute = self.attributes[idx]
        return sample, attribute
