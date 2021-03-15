import json
import os

import numpy as np

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from torchvision.datasets import ImageFolder

dataset_name = "Shapes"

class_attributes = {
    "Circle": [
        "n_edges_0",
        "n_angles_0",
        "angle_type_none",
    ],
    "Square": [
        "n_edges_4",
        "n_angles_4",
        "angle_type_right",
    ],
    "Triangle": [
        "n_edges_3",
        "n_angles_3",
        "angle_type_acute"
    ]
}


def download_shapes(force=False):
    if os.path.isdir("Shapes") and not force:
        print("Dataset already downloaded")
        return

    # Dataset need to be manually downloaded and moved to the data folder (this folder).
    # It will be a .zip file named dataset.zip
    # https://data.mendeley.com/datasets/wzr2yv7r53/1/files/72256ffc-4020-47e2-8ead-46b40fa15526

    os.system("tar -zxvf dataset.zip")
    os.remove("dataset.zip")
    os.rename("output", dataset_name)
    print(f"\nDataset {dataset_name} extracted")

    os.chdir(dataset_name)
    filenames = os.listdir(".")
    cls_files = {}
    for filename in filenames:
        cls = filename.split("_")[0]
        if cls in cls_files:
            cls_files[cls].append(filename)
        else:
            cls_files.update({cls: [filename]})
    print("Filenames read")

    for cls in cls_files.keys():
        if cls in class_attributes.keys():
            os.mkdir(cls)
            os.system(f"mv {cls}* {cls}")
        else:
            os.system(f"rm {cls}*")
    print("Dataset image folder structure created")

    dataset = ImageFolder(".")

    attributes = []
    for img in dataset.imgs:
        target = dataset.classes[img[1]]
        attributes.append(class_attributes[target])
    encoder = OneHotEncoder(sparse=False)
    attributes_one_hot = encoder.fit_transform(attributes)
    attributes_names = [name for cat in encoder.categories_ for name in cat]
    print(f"Attributes created ({attributes_one_hot.shape})")
    print(f"Names ({len(attributes_names)})")

    np.save("attributes.npy", attributes_one_hot)
    with open("attributes_names.txt", "w") as f:
        json.dump(attributes_names, f)
    print("Attributes saved")


if __name__ == "__main__":
    download_shapes()

