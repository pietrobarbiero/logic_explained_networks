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
    def __init__(self, root: str, transform: transforms, dataset_name: str, samples: list = None):
        super().__init__(root, transform)
        self.attributes = np.load(os.path.join(root, "attributes.npy"))
        self.attributes = self.attributes.astype(np.float32)
        # with open(os.path.join(root, "attribute_groups.json"), "r") as f:
        #     self.attribute_groups = json.load(f)
        # for attribute_group in self.attribute_groups.values():
        #     attribute_group_values = self.attributes[:, np.asarray(attribute_group)]
        #     occurrences = np.sum(attribute_group_values)
        #     assert occurrences <= dataset_length, "Wrong attribute groups"
        self.n_attributes = self.attributes.shape[1]
        self.dataset_name = dataset_name
        if samples is None:
            samples = [*range(len(self.imgs))]
        self.indices = samples

    def __getitem__(self, item):
        idx = self.indices[item]
        sample, _ = super().__getitem__(idx)
        attribute = self.attributes[idx]
        return sample, attribute

    def __len__(self):
        return len(self.indices)

