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

random.seed(2)
paths = get_custom_file_paths('/home/shared/Data/DRAC22/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set', '.png')
indices = list(range(len(paths)))
random.shuffle(indices)

data_root = '/home/lkreitner/OCTA-seg/datasets/'
mkdir(data_root)
data_root = '/home/lkreitner/OCTA-seg/datasets/DRAC_C'
mkdir(data_root)

train_split = 0.8
num_train_sample = int(len(indices)*0.8)

with open(os.path.join(data_root, 'train3.txt'), 'w+') as f:
    f.writelines([f"{i}\n" for i in indices[:num_train_sample]])
with open(os.path.join(data_root, 'val3.txt'), 'w+') as f:
    f.writelines([f"{i}\n" for i in indices[num_train_sample:]])
