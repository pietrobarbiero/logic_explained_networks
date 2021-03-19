import os

import torch
import torchvision
import numpy as np
import json

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
# from PIL import Image


def download_mnist(root="MNIST_EVEN_ODD", force=True):

    if os.path.isdir(root) and not force:
        print("Dataset already downloaded")
        return

    os.system(f"mkdir {root}")
    orig_dir = os.curdir
    os.chdir(root)

    # if not os.path.isfile("mnist.zip"):
    #     os.system("curl https://data.deepai.org/mnist.zip -o mnist.zip")
    #     print("Dataset Downloaded")
    # os.system("tar -zxvf mnist.zip")
    # os.remove("mnist.zip")
    # os.system("gzip -d *.gz")
    # os.system(f"mkdir {root}")
    # os.chdir(root)
    # os.system("mkdir raw")
    # os.system("mkdir processed")
    # os.chdir("..")
    # os.system(f"mv *ubyte {root}/raw")
    dataset = torchvision.datasets.MNIST(root=".", download=True)
    os.system("rm -r *")
    print("Dataset Extracted")

    j = 0
    attributes = np.zeros((len(dataset)*2, 10))
    transform = transforms.ToPILImage()
    for i, number in enumerate(dataset.classes):
        cls = "Even" if i % 2 == 0 else "Odd"
        if cls == "Even":
            if not os.path.isdir(cls):
                os.mkdir(cls)
            train_idx = torch.where(dataset.train_labels == i)[0].numpy()
            test_idx = torch.where(dataset.test_labels == i)[0].numpy()
            for idx in train_idx:
                im = transform(dataset.train_data[idx])
                im.save(os.path.join(cls, f"{j:07}.jpg"))
                attributes[j, i] = 1
                print(f"{i} class {j:07} image")
                j += 1

            for idx in test_idx:
                im = transform(dataset.test_data[idx])
                im.save(os.path.join(cls, f"{j:07}.jpg"))
                attributes[j, i] = 1
                print(f"{i} class {j:07} image")
                j += 1

    for i, number in enumerate(dataset.classes):
        cls = "Even" if i % 2 == 0 else "Odd"
        if cls == "Odd":
            if not os.path.isdir(cls):
                os.mkdir(cls)
            train_idx = torch.where(dataset.train_labels == i)[0].numpy()
            test_idx = torch.where(dataset.test_labels == i)[0].numpy()
            for idx in train_idx:
                im = transform(dataset.train_data[idx])
                im.save(os.path.join(cls, f"{j:07}.jpg"))
                attributes[j, i] = 1
                print(f"{i} class {j:07} image")
                j += 1

            for idx in test_idx:
                im = transform(dataset.test_data[idx])
                im.save(os.path.join(cls, f"{j:07}.jpg"))
                attributes[j, i] = 1
                print(f"{i} class {j:07} image")
                j += 1

    print("Dataset saved with Image Folder structure")

    np.save("attributes.npy", attributes)
    print("Attributes saved")

    attributes_names = [c[4:].title() for c in dataset.classes]
    with open("attributes_names.txt", "w") as f:
        json.dump(attributes_names, f)

    os.chdir(orig_dir)


if __name__ == "__main__":
    download_mnist()

