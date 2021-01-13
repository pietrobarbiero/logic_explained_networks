import os
import numpy as np

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
os.system("mv ../attributes.txt .")
os.rename("attributes.txt", "attributes_names.txt")
for item in os.listdir():
    if item != "images":
        os.system(f"rm -r {item}")
os.system("mv images/* .")
os.system("rm -r images")
print("Dataset configured")
