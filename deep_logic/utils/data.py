import json
import os
import random

import numpy as np
from typing import Tuple


from data.i2c_dataset import ImageToConceptDataset


def get_splits_train_val_test(dataset: ImageToConceptDataset, val_split: float = 0.1, test_split: float = 0.1) \
        -> Tuple[ImageToConceptDataset, ImageToConceptDataset, ImageToConceptDataset]:
    train_json = os.path.join(os.path.dirname(dataset.root), f"train_samples_{dataset.dataset_name}.json")
    val_json = os.path.join(os.path.dirname(dataset.root), f"val_samples_{dataset.dataset_name}.json")
    test_json = os.path.join(os.path.dirname(dataset.root), f"test_samples_{dataset.dataset_name}.json")

    # Creating dataset for Validation by splitting the samples in the dataset
    if os.path.isfile(val_json):
        with open(os.path.join(val_json), "r") as f:
            val_file = json.load(f)
            val_samples = val_file["samples"]
    else:
        dataset.targets = np.array(dataset.targets)
        val_len = {c: int(round(np.sum(dataset.targets == dataset.class_to_idx[c]) * val_split)) 
                   for c in dataset.classes}
        val_samples = []
        for c in dataset.classes:
            indices = np.argwhere(dataset.targets == dataset.class_to_idx[c]).squeeze()
            val_samples.extend(sorted(random.sample(indices.tolist(), val_len[c])))
        with open(os.path.join(val_json), "w") as f:
            val_file = {"samples": val_samples}
            json.dump(val_file, f)
    val_dataset = ImageToConceptDataset(dataset.root, dataset.transform, dataset.dataset_name, 
                                        samples=val_samples)

    # Creating dataset for Validation by splitting the samples in the dataset
    if os.path.isfile(test_json):
        with open(os.path.join(test_json), "r") as f:
            test_file = json.load(f)
            test_samples = test_file["samples"]
    else:
        dataset.targets = np.array(dataset.targets)
        test_len = {c: int(round(np.sum(dataset.targets == dataset.class_to_idx[c]) * test_split)) 
                    for c in dataset.classes}
        test_samples = []
        for c in dataset.classes:
            indices = np.argwhere(dataset.targets == dataset.class_to_idx[c]).squeeze()
            indices = [int(i) for i in indices if i not in val_samples]
            test_samples.extend(sorted(random.sample(indices, test_len[c])))
        with open(os.path.join(test_json), "w") as f:
            test_file = {"samples": test_samples}
            json.dump(test_file, f)
    test_dataset = ImageToConceptDataset(dataset.root, dataset.transform, dataset.dataset_name, 
                                         samples=test_samples)

    # Creating dataset for Training with the remaining samples
    if os.path.isfile(train_json):
        with open(os.path.join(train_json), "r") as f:
            train_file = json.load(f)
            train_samples = train_file["samples"]
    else:
        train_samples = [i for i in dataset.indices if i not in val_samples and i not in test_samples]
        with open(os.path.join(train_json), "w") as f:
            train_file = {"samples": train_samples}
            json.dump(train_file, f)
    train_dataset = ImageToConceptDataset(dataset.root, dataset.transform, dataset.dataset_name, 
                                          samples=train_samples)

    return train_dataset, val_dataset, test_dataset
