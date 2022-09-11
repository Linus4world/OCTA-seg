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

random.seed(12)
# paths = get_custom_file_paths('/home/shared/Data/DRAC22/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set', '.png')
paths = get_custom_file_paths('/home/shared/Data/DRAC22/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set', '.png')
indices = list(range(len(paths)))
random.shuffle(indices)

data_root = '/home/lkreitner/OCTA-seg/datasets/'
mkdir(data_root)
data_root = '/home/lkreitner/OCTA-seg/datasets/DRAC_C'
mkdir(data_root)

train_split = 0.8
num_train_sample = int(len(indices)*0.8)

with open(os.path.join(data_root, 'train.txt'), 'w+') as f:
    f.writelines([f"{i}\n" for i in indices[:num_train_sample]])
with open(os.path.join(data_root, 'val.txt'), 'w+') as f:
    f.writelines([f"{i}\n" for i in indices[num_train_sample:]])

labels_file = "/home/shared/Data/DRAC22/C. Diabetic Retinopathy Grading/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv"
reader = csv.reader(open(labels_file, 'r'))
next(reader)
all_labels = [int(v) for k, v in reader]
train_labels = list(np.array(all_labels)[indices[:num_train_sample]])

titles=["Normal (0)", "Non-proliferative (1)", "Proliferative (2)"]
d = {
    "Normal (0)": 0,
    "Non-proliferative (1)": 0,
    "Proliferative (2)": 0
}
for l in train_labels:
    d[titles[l]]+=1
plt.figure()
plt.title("C: Diabetic Retinopathy Grading - Training set")
plt.pie(d.values(), labels=d.keys(), autopct='%1.1f%%',shadow=True)
plt.savefig("/home/lkreitner/OCTA-seg/datasets/DRAC_C/distribution_train.png",  bbox_inches='tight')
plt.close()

val_labels = list(np.array(all_labels)[indices[num_train_sample:]])

titles=["Normal (0)", "Non-proliferative (1)", "Proliferative (2)"]
d = {
    "Normal (0)": 0,
    "Non-proliferative (1)": 0,
    "Proliferative (2)": 0
}
for l in val_labels:
    d[titles[l]]+=1
plt.figure()
plt.title("C: Diabetic Retinopathy Grading - Validation set")
plt.pie(d.values(), labels=d.keys(), autopct='%1.1f%%',shadow=True)
plt.savefig("/home/lkreitner/OCTA-seg/datasets/DRAC_C/distribution_val.png",  bbox_inches='tight')
plt.close()
