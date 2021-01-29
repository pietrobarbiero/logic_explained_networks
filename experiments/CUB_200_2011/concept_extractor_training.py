from datetime import datetime

import sklearn
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Subset

from data import data_transforms, CUB200
from data.i2c_dataset import ImageToConceptDataset
from deep_logic.utils import metrics
from deep_logic.utils.base import set_seed
from deep_logic.utils.data import get_splits_train_val_test, show_batch, get_splits_for_fsc
from image_preprocessing import cnn_models
from image_preprocessing.concept_extractor import CNNConceptExtractor

if __name__ == '__main__':

    root = "../../data/CUB_200_2011"
    epochs = 50
    seeds = [0]  # [0, 1, 2]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    metric = metrics.F1Score
    cnn_model = cnn_models.RESNET18
    pretrained = True
    transfer_learning = False  # True
    show_image = True
    data_augmentation = True
    few_shot = True
    denoised = True
    reduced = True
    l_r = 0.003
    batch_size = 64
    binary_loss = False
    if binary_loss:
        loss = torch.nn.BCELoss
    else:
        loss = torch.nn.NLLLoss

    for seed in seeds:
        set_seed(seed)
        train_transform = data_transforms.get_transform(dataset=CUB200, data_augmentation=data_augmentation,
                                                        inception=cnn_model == cnn_models.INCEPTION)
        test_transform = data_transforms.get_transform(dataset=CUB200, data_augmentation=False,
                                                       inception=cnn_model == cnn_models.INCEPTION)
        # dataset = torchvision.datasets.ImageFolder(root, train_transform)
        dataset = ImageToConceptDataset(root, train_transform, dataset_name=CUB200, denoised=denoised)
        if reduced:
            bill_attributes = 8 if reduced else 9
            dataset.attributes = dataset.attributes[:, :8]
            dataset.attribute_names = dataset.attribute_names[:8]
            dataset.n_attributes = 8
        if few_shot:
            train_set, val_set = get_splits_for_fsc(dataset, test_transform=test_transform)
            test_idx = train_set.indices + val_set.indices
            test_set = Subset(val_set.dataset, test_idx)
        else:
            train_set, val_set, test_set = get_splits_train_val_test(dataset, test_transform=test_transform)

        if show_image:
            show_batch(train_set, train_set.dataset.attribute_names)
            show_batch(val_set, val_set.dataset.attribute_names)
            show_batch(test_set, test_set.dataset.attribute_names)

        # name = f"model_{cnn_model}_prtr_{pretrained}_trlr_{transfer_learning}_bl_{binary_loss}_fs_{few_shot}_dataset_" \
        #        f"{CUB200}_denoised_{denoised}_reduced_{reduced}_lr_{l_r}_epochs_{epochs}_seed_{seed}_" \
        #        f"time_{datetime.now().strftime('%d-%m-%y %H:%M:%S')}"
        name = "model_Resnet_18_prtr_True_trlr_False_bl_False_fs_True_dataset_cub200_denoised_True_reduced_True_" \
               "lr_0.003_epochs_50_seed_0_time_28-01-21 16:41:10"
        print(name)
        model = CNNConceptExtractor(dataset.n_attributes, cnn_model=cnn_model,
                                    loss=loss(), name=name, pretrained=pretrained, transfer_learning=transfer_learning)
        # It takes a few
        # model.load(device)
        # results = model.fit(train_set=train_set, val_set=val_set, epochs=epochs, num_workers=8, l_r=l_r,
        #                     lr_scheduler=True, device=device, metric=metric(), batch_size=batch_size)
        # results.to_csv("results_" + name + ".csv")

        val = model.evaluate(val_set, metric=metric(), num_workers=8)
        print("Performance on val set:", val)

        val = model.evaluate(test_set, metric=metric(), num_workers=8)
        print("Performance on test set:", val)
