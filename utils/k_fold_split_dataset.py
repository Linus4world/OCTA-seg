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
stratified = False
train_split = 1-(1/K)

random.seed(0)
# data_root = '/home/lkreitner/OCTA-seg/datasets/'
# mkdir(data_root)
data_root = '/home/lkreitner/OCTA-seg/datasets/Edinburgh/original_images'
mkdir(data_root)


if stratified:
    labels_file = "/home/shared/Data/DRAC22/B. Image Quality Assessment/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv"
    reader = csv.reader(open(labels_file, 'r'))
    next(reader)
    all_labels = np.array([int(v) for k, v in reader])
    all_indices=list(range(len(all_labels)))
    indices_0 = np.where(all_labels==0)[0]
    indices_1 = np.where(all_labels==1)[0]
    indices_2 = np.where(all_labels==2)[0]
    random.shuffle(indices_0)
    random.shuffle(indices_1)
    random.shuffle(indices_2)

    start_ind_0 = 0
    end_ind_0 = 0
    start_ind_1 = 0
    end_ind_1 = 0
    start_ind_2 = 0
    end_ind_2 = 0
    for i in range(K):
        start_ind_0 = end_ind_0
        end_ind_0 = round(((i+1)/K) * len(indices_0))
        start_ind_1 = end_ind_1
        end_ind_1 = round(((i+1)/K) * len(indices_1))
        start_ind_2 = end_ind_2
        end_ind_2 = round(((i+1)/K) * len(indices_2))

        val_split_i = np.sort([
            *indices_0[start_ind_0:end_ind_0],
            *indices_1[start_ind_1:end_ind_1],
            *indices_2[start_ind_2:end_ind_2]
        ])
        train_split_i = [l for l in all_indices if l not in val_split_i]

        with open(os.path.join(data_root, f'train_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in train_split_i])
        with open(os.path.join(data_root, f'val_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in val_split_i])
else:
    paths = get_custom_file_paths(data_root, '')
    all_indices = list(range(len(paths)))
    random.shuffle(all_indices)
    start_ind = None
    end_ind = 0
    for i in range(K):
        start_ind = end_ind
        end_ind = round(((i+1)/K) * len(all_indices))
        val_split_i = np.sort(all_indices[start_ind:end_ind])
        train_split_i = np.sort([l for l in all_indices if l not in val_split_i])

        with open(os.path.join(data_root, f'train_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in train_split_i])
        with open(os.path.join(data_root, f'val_{i}.txt'), 'w+') as f:
            f.writelines([f"{i}\n" for i in val_split_i])

