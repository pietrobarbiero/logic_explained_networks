import os
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder


class ConceptToTaskDataset(ImageFolder):
    """
    Simple Dataset to use for Concept extraction that extend ImageFolder dataset from torchvision.
    Instead of returning the single class the image belongs to, it returns the array of visible attributes on the image.

    :param root: path to the main folder of the dataset
    """
    def __init__(self, root: str, dataset_name: str = "CUB200", samples: list = None):
        super().__init__(root)
        self.attributes = np.load(os.path.join(root, "attributes.npy"))
        self.attribute_names = pd.read_csv(os.path.join(root, "attributes_names.txt"), sep=" ", header=None)
        self.attribute_names = self.attribute_names.iloc[:, 1].values
        self.attributes = self.attributes.astype(np.float32)
        self.n_attributes = self.attributes.shape[1]
        self.dataset_name = dataset_name
        if samples is None:
            samples = [*range(len(self.imgs))]
        self.indices = samples

    def __getitem__(self, item):
        idx = self.indices[item]
        _, target = super().__getitem__(idx)
        attribute = self.attributes[idx]
        return attribute, target

    def __len__(self):
        return len(self.indices)

