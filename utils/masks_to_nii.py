import os
import cv2
import SimpleITK
import numpy as np


def read_nii(nii_path, data_type=np.uint16):
    img = SimpleITK.ReadImage(nii_path)
    data = SimpleITK.GetArrayFromImage(img)
    return np.array(data, dtype=data_type)


def arr2nii(data, filename, reference_name=None):
    img = SimpleITK.GetImageFromArray(data)/255
    if (reference_name is not None):
        img_ref = SimpleITK.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    SimpleITK.WriteImage(img, filename)


def masks2nii(mask_path):
    name=mask_path.split("/")[-1]
    mask_name_list = os.listdir(mask_path)
    mask_name_list = sorted(mask_name_list, reverse=False, key=lambda x: int(x[:-4]))
    mask_list = []
    for mask_name in mask_name_list:
        mask = cv2.imread(os.path.join(mask_path, mask_name), -1)
        mask_list.append(mask)
    arr2nii(np.array(mask_list, np.uint8), os.path.join(mask_path,f"{name}.nii.gz"))


if __name__ == "__main__":
    path = "your predicted masks path"
    masks2nii(path)
