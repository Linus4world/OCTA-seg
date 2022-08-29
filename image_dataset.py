from monai.data import DataLoader, Dataset
from monai.transforms import *
from data_transforms import AddRealNoised, RandCropOrPadd, ToDict, Resized, AddLineArtifact
import os
from numpy import array, deg2rad
import csv
import torch

from monai.data.meta_obj import set_track_meta

from metrics import Task
set_track_meta(False)

def get_custom_file_paths(folder, name):
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            if filename.endswith(name):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths

def _get_transformation(config, task: Task, phase: str, dtype=torch.float32) -> Compose:
    """
    Create and return the data transformations for 2D segmentation images the given phase.
    """
    if task == Task.VESSEL_SEGMENTATION:
        if phase == "train":
            return Compose([
                LoadImage(image_only=True),
                ScaleIntensity(0, 1),
                AddChannel(),
                ToDict(),
                # AddRandomNoised(noise_layer_path=config["Data"]["noise_map_path"]),
                AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0,1]),
                RandCropOrPadd(keys=["image", "label"], prob=0.5, min_factor=0.5, max_factor=1.5),
                RandRotate90d(keys=["image", "label"], prob=.75),
                RandRotated(keys=["image", "label"], range_x=deg2rad(10)),
                Rand2DElasticd(keys=["image", "label"], prob=.5, spacing=(30,30), magnitude_range=(1,3), padding_mode='zeros'),
                # Resized(keys=["image", "label"], shape=(304,304)),
                Resized(keys=["image", "label"], shape=(1024,1024)),
                RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5,4.5)),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddLineArtifact(keys=["image"]),
                AsDiscreted(keys=["label"], threshold=0.001),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        elif phase == "validation":
            return Compose([
                LoadImage(image_only=True),
                ScaleIntensity(0, 1),
                AddChannel(),
                Rotate90(k=1),
                Flip(0),
                ToDict(),
                AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
                AsDiscreted(keys=["label"],threshold=0.001),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        else:
            return Compose([
                LoadImage(image_only=True),
                ScaleIntensity(0, 1),
                AddChannel(),
                Rotate90(k=1),
                Flip(0),
                CastToType(dtype=dtype)
            ])
    elif task == Task.RETINOPATHY_CLASSIFICATION:
        if phase == "train":
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                RandFlipd(keys=["image"], prob=.5, spatial_axis=[0, 1]),
                RandRotate90d(keys=["image"], prob=.75),
                RandRotated(keys=["image"], prob=1, range_x=deg2rad(10), padding_mode="zeros"),
                RandCropOrPadd(keys=["image"], prob=1, min_factor=0.7, max_factor=1.3),
                Resized(keys=["image"], shape=[1024,1024]),
                Rand2DElasticd(keys=["image"], prob=.5, spacing=(40,40), magnitude_range=(1,2), padding_mode='zeros'),
                RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.5,1.5)),
                # RandBiasFieldd(keys=["image"], prob=0.1, degree=3, coeff_range=(0.1,0.3)),
                RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=0),
                Rotate90d(keys=["image"], k=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=0),
                Rotate90d(keys=["image"], k=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
    elif task == Task.IMAGE_QUALITY_CLASSIFICATION:
        if phase == "train":
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                RandFlipd(keys=["image"], prob=.5, spatial_axis=[0, 1]),
                RandRotate90d(keys=["image"], prob=.75),
                # RandRotated(keys=["image"], prob=1, range_x=deg2rad(10), padding_mode="zeros"),
                # RandCropOrPadd(keys=["image"], prob=1, min_factor=0.7, max_factor=1.3),
                Resized(keys=["image"], shape=[1024,1024]),
                Rand2DElasticd(keys=["image"], prob=.5, spacing=(40,40), magnitude_range=(1,2), padding_mode='zeros'),
                RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.5,1.5)),
                # RandBiasFieldd(keys=["image"], prob=0.1, degree=3, coeff_range=(0.1,0.3)),
                RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=0),
                Rotate90d(keys=["image"], k=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=0),
                Rotate90d(keys=["image"], k=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
    elif task == Task.AREA_SEGMENTATION:
        if phase == "train":
            return Compose([
                LoadImaged(keys=["image", "label"], image_only=True),
                ScaleIntensityd(keys=["image", "label"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                RandFlipd(keys=["image", "label"], prob=.5, spatial_axis=[0, 1]),
                RandRotate90d(keys=["image", "label"], prob=.75),
                RandRotated(keys=["image", "label"], prob=1, range_x=deg2rad(10), padding_mode="zeros"),
                RandCropOrPadd(keys=["image", "label"], prob=1, min_factor=0.7, max_factor=1.3),
                Resized(keys=["image", "label"], shape=[1024,1024]),
                Rand2DElasticd(keys=["image", "label"], prob=.5, spacing=(40,40), magnitude_range=(1,2), padding_mode='zeros'),
                RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.7,1.3)),
                # RandBiasFieldd(keys=["image"], prob=0.1, degree=3, coeff_range=(0.1,0.3)),
                RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image", "label"], image_only=True),
                ScaleIntensityd(keys=["image", "label"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Flipd(keys=["image", "label"], spatial_axis=0),
                Rotate90d(keys=["image", "label"], k=1),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        else:
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=0),
                Rotate90d(keys=["image"], k=1),
                CastToTyped(keys=["image"], dtype=dtype)
            ])

def get_post_transformation(task: Task, num_classes=2) -> tuple[Compose]:
    """
    Create and return the data transformation that is applied to the model prediction before inference.
    """
    if task == Task.VESSEL_SEGMENTATION:
        return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), KeepLargestConnectedComponent()]), Compose([CastToType(dtype=torch.uint8)])
    elif task == task == Task.AREA_SEGMENTATION:
        return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]), Compose([CastToType(dtype=torch.uint8)])
    else:
        return Compose([Activations(softmax=True)]), Compose([AsDiscrete(to_onehot=num_classes)])

def get_dataset(config: dict, phase: str, batch_size=None) -> DataLoader:
    """
    Creates and return the dataloader for the given phase.
    """
    task = config["General"]["task"]
    transform = _get_transformation(config, task, phase, dtype=torch.float16 if config["General"]["amp"] else torch.float32)

    if phase == "test":
        data_path = config["Test"]["dataset_images"]
    else:
        data_path = config["Data"]["dataset_images"]

    with open(config[phase.capitalize()]["dataset_path"], 'r') as f:
        lines = f.readlines()
        indices = [int(line.rstrip()) for line in lines]

    image_paths = get_custom_file_paths(*data_path)
    image_paths = array(image_paths)[indices].tolist()

    if task == Task.VESSEL_SEGMENTATION:
        train_files = image_paths
    elif task == Task.RETINOPATHY_CLASSIFICATION or task == Task.IMAGE_QUALITY_CLASSIFICATION:
        if config["Data"]["dataset_labels"] != '':
            reader = csv.reader(open(config["Data"]["dataset_labels"], 'r'))
            next(reader)
            labels = [int(v) for k, v in reader]
            labels = list(array(labels)[indices])
            train_files = [{"image": img, "path": img, "label": torch.tensor(label)} for img, label in zip(image_paths, labels)]
        else:
            train_files = [{"image": img, "path": img} for img in image_paths]
    elif task == Task.AREA_SEGMENTATION:
        placeholder_path = os.path.join("/".join(config[phase.capitalize()]["dataset_path"].split('/')[:-1]),'placeholder.png')
        if config["Data"]["dataset_labels"]:
            train_files = []
            for img_path in image_paths:
                name = img_path.split('/')[-1]
                labels = []
                for path in config["Data"]["dataset_labels"]:
                    label_path = os.path.join(path, name)
                    if os.path.exists(label_path):
                        labels.append(label_path)
                    else:
                        labels.append(placeholder_path)
                train_files.append({"image": img_path, "path": img_path, "label": labels})
        else:
            train_files = [{"image": img, "path": img} for img in image_paths]


    data_set = Dataset(train_files, transform=transform)
    loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()]["batch_size"], shuffle=phase!="test", num_workers=4, pin_memory=torch.cuda.is_available())
    return loader
