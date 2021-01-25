import torch
import torchvision

from data import data_transforms, CUB200
from data.i2c_dataset import ImageToConceptDataset
from deep_logic.utils import metrics
from deep_logic.utils.base import set_seed
from deep_logic.utils.data import get_splits_train_val_test, show_batch
from image_preprocessing import cnn_models
from image_preprocessing.concept_extractor import CNNConceptExtractor

if __name__ == '__main__':

    root = "../../data/CUB_200_2011"
    # dataset = torchvision.datasets.ImageFolder(root, data_transforms.get_transform(CUB200))
    # show_batch(dataset, dataset.classes)
    epochs = 400
    seeds = [0]  # [0, 1, 2]
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    metric = metrics.F1Score
    cnn_model = cnn_models.RESNET18
    pretrained = True
    transfer_learning = False  # True
    show_image = False
    l_r = 0.001
    batch_size = 64
    loss = torch.nn.BCELoss
    print("\nepochs", epochs, "\ndevice", device, "\nmetric", metric, "\ncnn_model", cnn_model, "\npretrained",
          pretrained, "\ntransfer_learning", transfer_learning, "\nl_r", l_r, "\nloss", loss)

    for seed in seeds:
        print("\nseed", seed)
        set_seed(seed)
        train_transform = data_transforms.get_transform(dataset=CUB200, data_augmentation=True,
                                                        inception=cnn_model == cnn_models.INCEPTION)
        val_transform = data_transforms.get_transform(dataset=CUB200, data_augmentation=False,
                                                      inception=cnn_model == cnn_models.INCEPTION)

        dataset = ImageToConceptDataset(root, train_transform, dataset_name=CUB200)
        train_set, val_set, test_set = get_splits_train_val_test(dataset, val_transform)

        if show_image:
            show_batch(train_set, train_set.attribute_names)
            show_batch(val_set, val_set.attribute_names)
            show_batch(test_set, test_set.attribute_names)

        name = f"model_{cnn_model}_pr_tr_{pretrained}_tr_lr_{transfer_learning}_dataset_{CUB200}_lr_{l_r}_epochs_{epochs}_seed_{seed}"
        print(name)
        model = CNNConceptExtractor(n_classes=dataset.n_attributes, cnn_model=cnn_model,
                                    loss=loss(), name=name, pretrained=pretrained, transfer_learning=transfer_learning)
        # It takes a few
        results = model.fit(train_set=train_set, val_set=val_set, epochs=epochs, num_workers=4, l_r=l_r,
                            lr_scheduler=True, device=device, metric=metric(), batch_size=batch_size)
        assert results.shape == (epochs, 4)
        results.to_csv("results_" + name + ".csv")

        metric = model.evaluate(test_set, metric=metric())
        print("Performance on test set:", metric)
