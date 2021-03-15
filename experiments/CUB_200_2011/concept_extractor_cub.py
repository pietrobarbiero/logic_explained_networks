import sys
import os

sys.path.append(os.path.join('..', '..'))

import torch
import numpy as np
from torch.utils.data import Subset
from data import CUB200
from deep_logic.utils.datasets import ImageToConceptDataset
from deep_logic.utils import metrics
from deep_logic.utils.base import set_seed, ClassifierNotTrainedError
from deep_logic.utils.data import get_transform, get_splits_train_val_test, get_splits_for_fsc, show_batch
from deep_logic.concept_extractor import cnn_models
from deep_logic.concept_extractor.concept_extractor import CNNConceptExtractor


def concept_extractor_cub(dataset_root="..//..//data//CUB_200_2011", epochs=200, seeds=None,
                          device=torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu"),
                          metric=metrics.F1Score, cnn_model=cnn_models.RESNET18, pretrained=True,
                          transfer_learning=False, show_image=True, data_augmentation=True, few_shot=False,
                          denoised=True, reduced=False, l_r=0.003, batch_size=64, binary_loss=True
                          ):
    if seeds is None:
        seeds = [0]
    if binary_loss:
        loss = torch.nn.BCELoss
    else:
        loss = torch.nn.NLLLoss

    for seed in seeds:
        set_seed(seed)
        train_transform = get_transform(dataset=CUB200, data_augmentation=data_augmentation,
                                        inception=cnn_model == cnn_models.INCEPTION)
        test_transform = get_transform(dataset=CUB200, data_augmentation=False,
                                       inception=cnn_model == cnn_models.INCEPTION)
        dataset = ImageToConceptDataset(dataset_root, train_transform, dataset_name=CUB200, denoised=denoised)
        if reduced:
            bill_attributes = 8 if denoised else 9
            dataset.attributes = dataset.attributes[:, :bill_attributes]
            dataset.attribute_names = dataset.attribute_names[:bill_attributes]
            dataset.n_attributes = bill_attributes
        if few_shot:
            train_set, val_set = get_splits_for_fsc(dataset, train_split=0.8, test_transform=test_transform)
            test_idx = train_set.indices + val_set.indices
            test_set = Subset(val_set.dataset, test_idx)
        else:
            train_set, val_set, test_set = get_splits_train_val_test(dataset, test_transform=test_transform)
        print("Number of attributes", dataset.n_attributes)

        if show_image:
            show_batch(train_set, train_set.dataset.attribute_names)
            show_batch(val_set, val_set.dataset.attribute_names)
            show_batch(test_set, test_set.dataset.attribute_names)

        name = f"model_{cnn_model}_prtr_{pretrained}_trlr_{transfer_learning}_bl_{binary_loss}_fs_{few_shot}_dataset_" \
               f"{CUB200}_denoised_{denoised}_reduced_{reduced}_lr_{l_r}_epochs_{epochs}_seed_{seed}_" \
               f"time_{datetime.now().strftime('%d-%m-%y %H:%M:%S')}.pth"
        # name = "model_Resnet_18_prtr_True_trlr_False_bl_True_fs_False_dataset_cub200_denoised_True_reduced_False_" \
        #        "lr_0.003_epochs_200_seed_0_time_04-02-21 16:20:27.pth"
        print(name)
        model = CNNConceptExtractor(dataset.n_attributes, cnn_model=cnn_model,
                                    loss=loss(), name=name, pretrained=pretrained, transfer_learning=transfer_learning)
        try:
            model.load(device)
        except ClassifierNotTrainedError:
            # It takes a few
            results = model.fit(train_set=train_set, val_set=val_set, epochs=epochs, num_workers=8, l_r=l_r,
                                lr_scheduler=True, device=device, metric=metric(), batch_size=batch_size)
            results.to_csv("results_" + name + ".csv")

        with torch.no_grad():
            model.eval()
            preds, labels = model.predict(dataset, num_workers=8, device=device)
            val = model.evaluate(dataset, metric=metric(), device=device, outputs=preds, labels=labels)
            np.save(os.path.join(dataset_root, f"{CUB200}_predictions.npy"), preds.cpu().numpy())
            print("Performance:", val)


if __name__ == '__main__':
    concept_extractor_cub()
