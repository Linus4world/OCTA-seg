import os
import random

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
paths = get_custom_file_paths('/home/lkreitner/arterial-tree-generation/dataset_22_07_22', 'gray_z.png')
indices = list(range(len(paths)))
random.shuffle(indices)

data_root = '/home/lkreitner/OCTA-seg/datasets/'
mkdir(data_root)
data_root = '/home/lkreitner/OCTA-seg/datasets/synth_24_07_22'
mkdir(data_root)

train_split = 0.8
num_train_sample = int(len(indices)*0.8)

with open(os.path.join(data_root, 'train.txt'), 'w+') as f:
    f.writelines([f"{i}\n" for i in indices[:num_train_sample]])
with open(os.path.join(data_root, 'val.txt'), 'w+') as f:
    f.writelines([f"{i}\n" for i in indices[num_train_sample:]])
