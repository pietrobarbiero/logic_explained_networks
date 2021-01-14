import torch

from data import data_transforms, CUB200
from data.i2c_dataset import ImageToConceptDataset
from deep_logic.utils import metrics
from deep_logic.utils.base import set_seed
from deep_logic.utils.data import get_splits_train_val_test
from image_preprocessing import cnn_models
from image_preprocessing.concept_extractor import CNNConceptExtractor

if __name__ == '__main__':

    epochs = 50
    seeds = [0, 1, 2]
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    metric = metrics.F1Score
    cnn_model = cnn_models.INCEPTION
    pretrained = True
    l_r = 0.01
    loss = torch.nn.BCELoss
    print("epochs", epochs, "\ndevice", device, "\nmetric", metric, "\ncnn_model", cnn_model, "\npretrained",
          pretrained, "\nl_r", l_r, "\nloss", loss)

    for seed in seeds:
        print("seed", seed)
        set_seed(seed)
        transform = data_transforms.get_transform(dataset=data_transforms.CUB200, data_augmentation=True)

        dataset = ImageToConceptDataset("../../data/CUB_200_2011", transform, dataset_name=CUB200)
        train_set, val_set, test_set = get_splits_train_val_test(dataset)

        name = f"model_{cnn_model}_dataset_{CUB200}_lr_{l_r}_epochs_{epochs}_seed_{seed}"
        model = CNNConceptExtractor(n_classes=dataset.n_attributes,
                                    cnn_model=cnn_model, loss=loss(), name=name, pretrained=pretrained)
        # It takes a few
        results = model.fit(train_set=train_set, val_set=val_set, epochs=epochs, l_r=l_r,
                            device=device, metric=metric())
        assert results.shape == (epochs, 4)
        results.to_csv("results_" + name + ".csv")

        metric = model.evaluate(test_set, metric=metric())
        print("Performance on test set:", metric)
