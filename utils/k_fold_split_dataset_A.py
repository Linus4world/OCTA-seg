import os
import random
import csv
import numpy as np

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

K = 5
stratified = True
train_split = 0.8

random.seed(0)
data_root = '/home/lkreitner/OCTA-seg/datasets/'
mkdir(data_root)
data_root = '/home/lkreitner/OCTA-seg/datasets/DRAC_A'
mkdir(data_root)

image_paths = get_custom_file_paths("/home/shared/Data/DRAC22/A. Segmentation/1. Original Images/a. Training Set",".png")
image_paths = [p.split("/")[-1] for p in image_paths]
all_indices = list(range(len(image_paths)))

if stratified:
    image_paths_0 = [p.split("/")[-1] for p in get_custom_file_paths("/home/shared/Data/DRAC22/A. Segmentation/2. Groundtruths/a. Training Set/1. Intraretinal Microvascular Abnormalities", ".png")]
    indices_0 = np.where([True if p in image_paths_0 else False for p in image_paths])[0]
    image_paths_1 = [p.split("/")[-1] for p in get_custom_file_paths("/home/shared/Data/DRAC22/A. Segmentation/2. Groundtruths/a. Training Set/2. Nonperfusion Areas", ".png")]
    indices_1 = np.where([True if p in image_paths_1 else False for p in image_paths])[0]
    image_paths_2 = [p.split("/")[-1] for p in get_custom_file_paths("/home/shared/Data/DRAC22/A. Segmentation/2. Groundtruths/a. Training Set/3. Neovascularization", ".png")]
    indices_2 = np.where([True if p in image_paths_2 else False for p in image_paths])[0]

    random.shuffle(indices_0)
    random.shuffle(indices_1)
    random.shuffle(indices_2)

    m = {i: [] for i in all_indices}
    for i in indices_0:
        m[i].append(0)
    for i in indices_1:
        m[i].append(1)
    for i in indices_2:
        m[i].append(2)

    splits = [[] for _ in range(5)]
    pointer = 0
    for i in [j for j in m if len(m[j])==3]:
        splits[pointer].append(i)
        pointer = (pointer+1)%5

    for i in [j for j in m if m[j]==[0,2]]:
        splits[pointer].append(i)
        pointer = (pointer+1)%5
    for i in [j for j in m if m[j]==[1,2]]:
        splits[pointer].append(i)
        pointer = (pointer+1)%5

    for i in [j for j in m if m[j]==[2]]:
        splits[pointer].append(i)
        pointer = (pointer+1)%5

    for i in [j for j in m if m[j]==[0,1]]:
        splits[pointer].append(i)
        pointer = (pointer+1)%5

    for i in [j for j in m if m[j]==[1]]:
        splits[pointer].append(i)
        pointer = (pointer+1)%5

    for i in [j for j in m if m[j]==[0]]:
        splits[pointer].append(i)
        pointer = (pointer+1)%5

    for i in range(K):
        val_split_i = np.sort(splits[i])
        train_split_i =  np.sort([l for l in all_indices if l not in val_split_i])

        with open(os.path.join(data_root, f'train_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in train_split_i])
        with open(os.path.join(data_root, f'val_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in val_split_i])
else:
    random.shuffle(all_indices)
    start_ind = None
    end_ind = 0
    for i in range(K):
        start_ind = end_ind
        end_ind = ((i+1)/K) * len(all_indices)
        val_split_i = np.sort(all_indices[start_ind:end_ind])
        train_split_i = np.sort([l for l in all_indices if l not in val_split_i])

        with open(os.path.join(data_root, f'train_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in train_split_i])
        with open(os.path.join(data_root, f'val_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in val_split_i])

