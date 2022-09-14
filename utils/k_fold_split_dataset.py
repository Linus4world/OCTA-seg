import os
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_custom_file_paths(folder, name):
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            if filename.endswith(name):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

random.seed(0)
data_root = '/home/lkreitner/OCTA-seg/datasets/'
mkdir(data_root)
data_root = '/home/lkreitner/OCTA-seg/datasets/DRAC_C'
mkdir(data_root)

labels_file = "/home/shared/Data/DRAC22/C. Diabetic Retinopathy Grading/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
reader = csv.reader(open(labels_file, 'r'))
next(reader)
all_labels = np.array([int(v) for k, v in reader])

indices_0 = np.where(all_labels==0)[0]
indices_1  = np.where(all_labels==1)[0]
indices_2  = np.where(all_labels==2)[0]

train_split = 0.8

for i in range(10):
    random.shuffle(indices_0)
    random.shuffle(indices_1)
    random.shuffle(indices_2)

    train_split_i = [
        *indices_0[:int(len(indices_0)*0.8)],
        *indices_1[:int(len(indices_1)*0.8)],
        *indices_2[:int(len(indices_2)*0.8)]
    ]
    val_split_i = [
        *indices_0[int(len(indices_0)*0.8):],
        *indices_1[int(len(indices_1)*0.8):],
        *indices_2[int(len(indices_2)*0.8):]
    ]
    with open(os.path.join(data_root, f'train_{i}.txt'), 'w+') as f:
        f.writelines([f"{i}\n" for i in train_split_i])
    with open(os.path.join(data_root, f'val_{i}.txt'), 'w+') as f:
        f.writelines([f"{i}\n" for i in val_split_i])

