import sys
import os
from datetime import datetime
from typing import List

sys.path.append(os.path.join('..', '..'))

import torch
import numpy as np
from torch.utils.data import Subset
from data import CUB200
from lens.utils.datasets import ImageToConceptDataset, ImageToConceptAndTaskDataset
from lens.utils import metrics
from lens.utils.base import set_seed, ClassifierNotTrainedError
from lens.utils.data import get_transform, get_splits_train_val_test, get_splits_for_fsc, show_batch
from lens.concept_extractor import cnn_models
from lens.concept_extractor.concept_extractor import CNNConceptExtractor
from lens.models.robust_cnn_classifier import RobustCNNClassifier


def concept_extractor_cub(dataset_root="..//data//CUB_200_2011", result_folder=".", epochs=200, seeds=None,
                          device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
                          metric=metrics.F1Score(), cnn_model=cnn_models.RESNET18, pretrained=True, multi_label=False,
                          transfer_learning=False, show_image=True, data_augmentation=True, few_shot=False,
                          reduced=False, l_r=0.003, batch_size=128, binary_loss=True, robust=False,
                          save_predictions=False) \
        -> List[RobustCNNClassifier]:

    dataset_name = CUB200
    if seeds is None:
        seeds = [0]

    if binary_loss:
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        loss = torch.nn.CrossEntropyLoss()

    models = []
    for seed in seeds:
        set_seed(seed)
        train_transform = get_transform(dataset=dataset_name, data_augmentation=data_augmentation,
                                        inception=cnn_model == cnn_models.INCEPTION)
        test_transform = get_transform(dataset=dataset_name, data_augmentation=False,
                                       inception=cnn_model == cnn_models.INCEPTION)
        if multi_label:
            dataset = ImageToConceptAndTaskDataset(dataset_root, train_transform, dataset_name=dataset_name)
        else:
            dataset = ImageToConceptDataset(dataset_root, train_transform, dataset_name=dataset_name)
        if reduced:
            bill_attributes = 8
            dataset.attributes = dataset.attributes[:, :bill_attributes]
            dataset.attribute_names = dataset.attribute_names[:bill_attributes]
            dataset.n_attributes = bill_attributes
        if few_shot:
            train_set, val_set = get_splits_for_fsc(dataset, train_split=0.8, test_transform=test_transform)
            test_idx = train_set.indices + val_set.indices
            test_set = Subset(val_set.dataset, test_idx)
        else:
            train_set, val_set, test_set = get_splits_train_val_test(dataset, test_transform=test_transform)
        print("Concept_extractor: " + "Number of attributes", dataset.n_attributes)

        if show_image:
            show_batch(train_set, train_set.dataset.attribute_names)
            show_batch(val_set, val_set.dataset.attribute_names)
            show_batch(test_set, test_set.dataset.attribute_names)

        name = f"model_{cnn_model}_robust_{robust}_prtr_{pretrained}_trlr_{transfer_learning}_da_{data_augmentation}" \
               f"_bl_{binary_loss}_fs_{few_shot}_mlb_{multi_label}_dataset_{dataset_name}_r_{reduced}" \
               f"_lr_{l_r}_epochs_{epochs}_seed_{seed}.pth"
        name_model = os.path.join(result_folder, name)
        print("Concept_extractor: " + name_model)

        n_classes = dataset.n_attributes
        main_classes = dataset.classes
        attributes_names = [a for a in dataset.attribute_names if a not in main_classes]
        if robust:
            model = RobustCNNClassifier(n_classes, main_classes, attributes_names, cnn_model, loss, transfer_learning,
                                        pretrained, name_model, device)
        else:
            model = CNNConceptExtractor(n_classes, cnn_model, loss, transfer_learning, pretrained, name_model, device)

        try:
            model.load(device)
        except ClassifierNotTrainedError:
            # It takes a few
            results = model.fit(train_set=train_set, val_set=val_set, epochs=epochs, num_workers=8, l_r=l_r,
                                lr_scheduler=True, device=device, metric=metric, batch_size=batch_size)
            results.to_csv(os.path.join(result_folder, "results_" + name) + ".csv")

        if save_predictions:
            with torch.no_grad():
                model.eval()
                preds, labels = model.predict(dataset, num_workers=8, batch_size=batch_size//8, device=device)
                val = model.evaluate(dataset, metric=metric, device=device, outputs=preds, labels=labels)
                if multi_label:
                    pred_path = os.path.join(dataset_root, f"{dataset_name}_multi_label_predictions.npy")
                else:
                    pred_path = os.path.join(dataset_root, f"{dataset_name}_predictions.npy")
                np.save(pred_path, preds.cpu().numpy())
                print("Concept_extractor: " + "Performance on the whole dataset:", val)

        models.append(model)

    return models


if __name__ == '__main__':
    concept_extractor_cub()
