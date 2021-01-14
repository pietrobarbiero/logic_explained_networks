import json
import os
import numpy as np


def download_cub(force=False):
    if os.path.isdir("CUB_200_2011") and not force:
        print("Dataset already downloaded")
        return

    os.system("git clone https://github.com/chentinghao/download_google_drive.git")
    os.system("python download_google_drive/download_gdrive.py 1hbzc_P1FuxMkcabkgn9ZKinBwW683j45 CUB_200_2011.tgz")
    print("Dataset downloaded")

    os.system("rm -r download_google_drive")
    os.system("tar -zxvf CUB_200_2011.tgz")
    os.remove("CUB_200_2011.tgz")
    print("Dataset extracted")

    os.chdir("CUB_200_2011")
    attributes = np.zeros(shape=(11788, 312))
    with open("attributes\\image_attribute_labels.txt") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(" ")
            if int(fields[2]) == 1:
                img_idx = int(fields[0]) - 1
                attr_idx = int(fields[1]) - 1
                attributes[img_idx][attr_idx] = 1
    np.save("attributes", attributes)
    print("Attribute saved")

    os.system("mv ../attributes.txt .")
    os.rename("attributes.txt", "attributes_names.txt")
    attribute_groups = {}
    with open("attributes_names.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(" ")
            attr_idx = int(fields[0]) - 1
            cur_prefix = fields[1].split(":")[0]
            if cur_prefix in attribute_groups:
                attribute_groups[cur_prefix].append(attr_idx)
            else:
                attribute_groups.update({cur_prefix: [attr_idx]})
    with open("attribute_groups.json", "w") as f:
        json.dump(attribute_groups, f)
    print("Attribute groups saved")

    for item in os.listdir():
        if item != "images":
            os.system(f"rm -r {item}")
    os.system("mv images/* .")
    os.system("rm -r images")
    print("Temporary file cleaned")
    print("Dataset configured")


if __name__ == "__main__":
    download_cub()
