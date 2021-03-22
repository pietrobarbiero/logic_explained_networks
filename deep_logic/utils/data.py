import copy
import json
import os
import random

import numpy as np
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from matplotlib import pyplot as plt
from .datasets import ConceptDataset

from data import CUB200, MNIST


def get_transform(dataset, data_augmentation=False, inception=False) -> transforms.Compose:
    size = 299 if inception else 224
    resize = int(size * 0.9)
    if dataset == CUB200:
        if data_augmentation:
            transform = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.CenterCrop(size=size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, hue=0.4, saturation=0.4),
                transforms.RandomRotation(0.4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=resize),
                transforms.CenterCrop(size=size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif dataset == MNIST:
        size = 299 if inception else 224
        resize = int(size * 0.9)
        transform = transforms.Compose([
            transforms.Resize(size=resize),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise NotImplementedError()
    return transform


def get_splits_train_val_test(dataset: ConceptDataset, val_split: float = 0.1, test_split: float = 0.1,
                              load=True, test_transform: transforms.Compose = None) \
        -> Tuple[Subset, Subset, Subset]:
    train_json = os.path.join(dataset.root, "..", f"train_samples_{dataset.dataset_name}.json")
    val_json = os.path.join(dataset.root, "..", f"val_samples_{dataset.dataset_name}.json")
    test_json = os.path.join(dataset.root, "..", f"test_samples_{dataset.dataset_name}.json")

    # Copying dataset if test_transform is given
    if test_transform is not None:
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.transform = test_transform
    else:
        dataset_copy = dataset

    # Checking dataset targets
    targets = np.asarray(dataset.targets)
    if len(targets.squeeze().shape) > 1:
        targets = targets.argmax(axis=1)

    # Creating dataset for Validation by splitting the samples in the dataset
    if os.path.isfile(val_json) and load:
        with open(os.path.join(val_json), "r") as f:
            val_file = json.load(f)
            val_samples = val_file["samples"]
    else:
        val_len = {c: int(round(np.sum(targets == dataset.class_to_idx[c]) * val_split))
                   for c in dataset.classes}
        val_samples = []
        for c in dataset.classes:
            indices = np.argwhere(targets == dataset.class_to_idx[c]).squeeze().tolist()
            val_samples.extend(sorted(random.sample(indices, val_len[c])))
        if load:
            with open(os.path.join(val_json), "w") as f:
                val_file = {"samples": val_samples}
                json.dump(val_file, f)
    val_dataset = Subset(dataset_copy, np.asarray(val_samples))

    # Creating dataset for Validation by splitting the samples in the dataset
    if os.path.isfile(test_json) and load:
        with open(os.path.join(test_json), "r") as f:
            test_file = json.load(f)
            test_samples = test_file["samples"]
    else:
        test_len = {c: int(round(np.sum(targets == dataset.class_to_idx[c]) * test_split))
                    for c in dataset.classes}
        test_samples = []
        for c in dataset.classes:
            indices = np.argwhere(targets == dataset.class_to_idx[c]).squeeze().tolist()
            indices = list(set(indices) - set(val_samples))  # [int(i) for i in indices if i not in val_samples]
            test_samples.extend(sorted(random.sample(indices, test_len[c])))
        if load:
            with open(os.path.join(test_json), "w") as f:
                test_file = {"samples": test_samples}
                json.dump(test_file, f)
    test_dataset = Subset(dataset_copy, np.asarray(test_samples))

    # Creating dataset for Training with the remaining samples
    if os.path.isfile(train_json) and load:
        with open(os.path.join(train_json), "r") as f:
            train_file = json.load(f)
            train_samples = train_file["samples"]
    else:
        train_samples = list({*range(len(dataset))} - set(val_samples) - set(test_samples))  # [i for i in range(len(dataset)) if i not in val_samples and i not in test_samples]
        if load:
            with open(os.path.join(train_json), "w") as f:
                train_file = {"samples": train_samples}
                json.dump(train_file, f)
    train_dataset = Subset(dataset, np.asarray(train_samples))

    return train_dataset, val_dataset, test_dataset


def get_splits_for_fsc(dataset: ConceptDataset, train_split: float = 0.5, load=False,
                       test_transform: transforms.Compose = None) -> Tuple[Subset, Subset]:
    train_json = os.path.join(os.path.dirname(dataset.root), f"samples_{dataset.dataset_name}_training.json")
    ft_json = os.path.join(os.path.dirname(dataset.root), f"samples_{dataset.dataset_name}_fine_tuning.json")

    # Copying dataset if test_transform is given
    if test_transform is not None:
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.transform = test_transform
    else:
        dataset_copy = dataset

    # Creating dataset for Fine-tuning by splitting the classes of the dataset
    if os.path.isfile(train_json) and load:
        with open(os.path.join(train_json), "r") as f:
            train_file = json.load(f)
            train_samples = train_file["samples"]
            train_classes = train_file["classes"]
    else:
        train_len_split = int(len(dataset.classes) * train_split)
        train_classes = sorted(random.sample(dataset.classes, train_len_split))
        train_samples = [i for i in range(len(dataset)) if dataset.classes[dataset.targets[i]] in train_classes]
        assert abs(len(train_samples) / len(dataset) - train_split) < 0.01, "Error while splitting the dataset"
        with open(os.path.join(train_json), "w") as f:
            train_file = {"samples": train_samples, "classes": train_classes}
            json.dump(train_file, f)
    train_dataset = Subset(dataset, train_samples)

    # Creating dataset for Training with the all the classes
    if os.path.isfile(ft_json) and load:
        with open(os.path.join(ft_json), "r") as f:
            ft_file = json.load(f)
            ft_samples = ft_file["samples"]
            ft_classes = ft_file["classes"]
    else:
        ft_samples = {*range(len(dataset))} - set(train_samples)  #  [i for i in range(len(dataset)) if i not in train_samples]
        ft_classes = [c for c in dataset.classes if c not in train_classes]
        with open(os.path.join(ft_json), "w") as f:
            ft_file = {"samples": ft_samples, "classes": ft_classes}
            json.dump(ft_file, f)
    assert abs(len(ft_classes) - len(dataset.classes) * (1 - train_split)) < 0.01, "Error while splitting dataset"
    ft_dataset = Subset(dataset_copy, ft_samples)

    return train_dataset, ft_dataset


def clean_names(names: list) -> list:
    names = \
        [
            name.replace("::", "_")
                .replace("\n", "")
                .replace("-", "")
                .replace("(", "")
                .replace(")", "")
                .replace(" ", "_")
            for name in names
        ]
    return names


def show_batch(dataset, labels_names, batch_size=8, save=False):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_data = next(iter(data_loader))
    assert len(batch_data) == 2, "Error when loading data"
    batch_images, batch_labels = batch_data[0], batch_data[1]
    fig = plt.figure()
    plt.rcParams.update({'font.size': 6})
    for j, sample in enumerate(batch_images):
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

        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        sample *= std
        sample += mean
        if np.max(sample) < 200 or np.min(sample) < 0:
            sample = (sample * 255).astype(np.uint8)

        ax = plt.subplot(batch_size / 2, 2, j + 1)
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
