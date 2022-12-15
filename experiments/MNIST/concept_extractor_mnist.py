import sys
import os

sys.path.append(os.path.join('..', '..'))

import torch
import numpy as np
from torch.utils.data import Subset
from data import MNIST
from lens.utils.datasets import ImageToConceptDataset, ImageToConceptAndTaskDataset, ImageToTaskDataset
from lens.utils import metrics
from lens.utils.base import set_seed, ClassifierNotTrainedError
from lens.utils.data import get_transform, get_splits_train_val_test, get_splits_for_fsc, show_batch
import cnn_models
from concept_extractor import CNNConceptExtractor
from lens.utils.metrics import Accuracy


def concept_extractor_MNIST(dataset_root=f"..//..//data//MNIST_EVEN_ODD", epochs=20, seeds=None, device=None,
                            metric=metrics.F1Score, cnn_model=cnn_models.RESNET10, pretrained=False, multi_label=False,
                            only_main=False, transfer_learning=False, eval_test=False, show_image=True,
                            save_predictions=False, data_augmentation=False,
                            few_shot=False, l_r=0.01, batch_size=128, binary_loss=False):

    assert not (only_main and multi_label), "Cannot impose both multi_label and main classes"

    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    dataset_name = MNIST
    if seeds is None:
        seeds = [0]

    if binary_loss:
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        loss = torch.nn.CrossEntropyLoss()

    models = []
    test_accs = []
    for seed in seeds:
        set_seed(seed)
        train_transform = get_transform(dataset=dataset_name, data_augmentation=data_augmentation,
                                        inception=cnn_model == cnn_models.INCEPTION)
        test_transform = get_transform(dataset=dataset_name, data_augmentation=False,
                                       inception=cnn_model == cnn_models.INCEPTION)
        if multi_label:
            dataset = ImageToConceptAndTaskDataset(dataset_root, train_transform, dataset_name=dataset_name)
        else:
            if only_main:
                dataset = ImageToTaskDataset(dataset_root, train_transform, dataset_name=dataset_name)
            else:
                dataset = ImageToConceptDataset(dataset_root, train_transform, dataset_name=dataset_name)

        if few_shot:
            train_set, val_set = get_splits_for_fsc(dataset, train_split=0.8, test_transform=test_transform)
            test_idx = train_set.indices + val_set.indices
            test_set = Subset(val_set.dataset, test_idx)
        else:
            train_set, val_set, test_set = get_splits_train_val_test(dataset, val_split=0.1,
                                                                     test_split=0.5, test_transform=test_transform)
        print("Number of attributes", dataset.n_attributes)

        if show_image:
            show_batch(train_set, train_set.dataset.attribute_names if not only_main else train_set.dataset.classes)
            show_batch(val_set, val_set.dataset.attribute_names if not only_main else train_set.dataset.classes)
            show_batch(test_set, test_set.dataset.attribute_names if not only_main else train_set.dataset.classes)

        name = f"model_{cnn_model}_prtr_{pretrained}_trlr_{transfer_learning}_bl_{binary_loss}_fs_{few_shot}_" \
               f"ml_{multi_label}_om_{only_main}_dataset_{dataset_name}_lr_{l_r}_epochs_{epochs}_seed_{seed}.pth" \
               # f"time_{datetime.now().strftime('%d-%m-%y %H:%M:%S')}.pth"
        print(name)
        model = CNNConceptExtractor(dataset.n_attributes, cnn_model=cnn_model,
                                    loss=loss, name=name, pretrained=pretrained, transfer_learning=transfer_learning)
        try:
            model.load(device)
        except ClassifierNotTrainedError:
            # It takes a few
            results = model.fit(train_set=train_set, val_set=val_set, epochs=epochs, num_workers=8, l_r=l_r,
                                lr_scheduler=True, device=device, metric=metric(), batch_size=batch_size)
            results.to_csv("results_" + name + ".csv")

        if eval_test:
            test_acc = model.evaluate(val_set, batch_size=1028, device=device)
            print("Test acc", test_acc)
            test_accs.append(test_acc)

        if save_predictions:
            with torch.no_grad():
                model.eval()
                preds, labels = model.predict(dataset, num_workers=8, device=device)
                val = model.evaluate(dataset, metric=metric(), device=device, outputs=preds, labels=labels)
                if multi_label:
                    pred_path = os.path.join(dataset_root, f"{dataset_name}_multi_label_predictions.npy")
                else:
                    pred_path = os.path.join(dataset_root, f"{dataset_name}_predictions.npy")
                np.save(pred_path, preds.cpu().numpy())
                print("Performance:", val)

        models.append(model)

    if eval_test:
        return models, np.asarray(test_accs)

    return models

if __name__ == '__main__':
    seeds = [0, 1, 2]
    metric = Accuracy()
    device = torch.device("cuda:0")  # torch.device("cpu")
    _, test_accs = concept_extractor_MNIST(device=device, only_main=True,
                                           seeds=seeds, eval_test=True,
                                           binary_loss=False, metric=metric)
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    print(f"Test acc {mean_test_acc} +- {std_test_acc}")
