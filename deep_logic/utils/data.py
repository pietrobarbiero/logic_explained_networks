import json
import os
import random

import numpy as np
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from data.i2c_dataset import ImageToConceptDataset
from matplotlib import pyplot as plt


def get_splits_train_val_test(dataset: ImageToConceptDataset, val_transform: transforms.Compose,
                              val_split: float = 0.1, test_split: float = 0.1) \
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
    val_dataset = ImageToConceptDataset(dataset.root, val_transform, dataset.dataset_name,
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
    test_dataset = ImageToConceptDataset(dataset.root, val_transform, dataset.dataset_name,
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


def show_batch(dataset, labels_names, batch_size=8, save=False):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, random=True)
    batch_data = next(iter(data_loader))
    assert len(batch_data) == 2, "Error when loading data"
    batch_images, batch_labels = batch_data[0], batch_data[1]
    fig = plt.figure()
    plt.rcParams.update({'font.size': 6})
    for j, sample in enumerate(batch_images):
        ax = plt.subplot(batch_size/2, 2, j + 1)
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        elif isinstance(sample, np.ndarray):
            sample = sample.squeeze()
        elif isinstance(sample, Image.Image):
            sample = np.asarray(sample)
        else:
            raise NotImplementedError
        if sample.shape[0] == 3:
            sample = np.rollaxis(sample, 0, 3)
        if np.max(sample) < 200 or np.min(sample) < 0:
            sample = (sample * 255).astype(np.uint8)
        plt.imshow(sample)
        title = ""
        label = batch_labels[j]
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if label.size > 1:
            for i in np.where(label == 1)[0]:
                title += labels_names[i] + ", "
        else:
            title = labels_names[label]
        title = title[:100]
        ax.set_title(title)
        ax.axis('off')
    if save:
        plt.savefig("../images/fig" + str(np.random.randint(1000)))
    plt.show()
