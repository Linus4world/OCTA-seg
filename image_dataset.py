from monai.data import DataLoader, Dataset
from monai.transforms import *
from data_transforms import AddRealNoised, ToDict, Resized, AddLineArtifact
import os
from numpy import deg2rad, array
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
    if task == Task.VESSEL_SEGMENTATION.value:
        if phase == "train":
            return Compose([
                LoadImage(image_only=True),
                ScaleIntensity(0, 1),
                AddChannel(),
                Rotate(deg2rad(270)),
                Flip(0),
                ToDict(),
                # AddRandomNoised(noise_layer_path=config["Data"]["noise_map_path"]),
                AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
                RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=[0,1]),
                RandScaleCropd(keys=["image", "label"], roi_scale=0.5, max_roi_scale=1, random_center=True, random_size=True),
                RandRotated(keys=["image", "label"], range_x=10),
                Resized(keys=["image", "label"], shape=(1024,1024)),
                AddLineArtifact(keys=["image"]),
                AsDiscreted(keys=["label"],threshold=0.1),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        elif phase == "validation":
            return Compose([
                LoadImage(image_only=True),
                ScaleIntensity(0, 1),
                AddChannel(),
                Rotate(deg2rad(270)),
                Flip(0),
                ToDict(),
                AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
                RandScaleCropd(keys=["image", "label"], roi_scale=0.3, max_roi_scale=1, random_center=True, random_size=True),
                Resized(keys=["image", "label"], shape=(1024,1024)),
                AddLineArtifact(keys=["image"]),
                AsDiscreted(keys=["label"],threshold=0.1),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        else:
            return Compose([
                LoadImage(image_only=True),
                ScaleIntensity(0, 1),
                AddChannel(),
                Rotate(deg2rad(270)),
                Flip(0),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
    else:
        if phase == "train":
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                RandFlipd(keys=["image"], prob=0.1, spatial_axis=[0, 1]),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                AddChanneld(keys=["image"]),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                AddChanneld(keys=["image"]),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])

def get_post_transformation(task: Task, num_classes=2) -> tuple[Compose]:
    """
    Create and return the data transformation that is applied to the model prediction before inference.
    """
    if task == Task.VESSEL_SEGMENTATION.value or task == Task.AREA_SEGMENTATION.value:
        return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]), Compose([])
    else:
        return Compose([Activations(softmax=True)]), Compose([AsDiscrete(to_onehot=num_classes)])

def get_dataset(config: dict, phase: str, batch_size=None) -> DataLoader:
    """
    Creates and return the dataloader for the given phase.
    """
    task = config["General"]["task"]
    transform = _get_transformation(config, task, phase, dtype=torch.float16 if config["General"]["amp"] else torch.float32)

    data_path = config["Data"]["dataset_images"]
    if phase == "test":
        data_path = config["Test"]["dataset_images"]

    with open(config[phase.capitalize()]["dataset_path"], 'r') as f:
        lines = f.readlines()
        indices = [int(line.rstrip()) for line in lines]
    if task == Task.VESSEL_SEGMENTATION.value:
        image_paths = get_custom_file_paths(*data_path)
        image_paths = list(array(image_paths)[indices])
        train_files = image_paths
    else:
        reader = csv.reader(open(config["Data"]["dataset_labels"], 'r'))
        next(reader)
        labels = [int(v) for k, v in reader]
        labels = list(array(labels)[indices])

        image_paths = get_custom_file_paths(*data_path)
        image_paths = list(array(image_paths)[indices])
        train_files = [{"image": img, "label": torch.tensor(label)} for img, label in zip(image_paths, labels)]
    data_set = Dataset(train_files, transform=transform)
    loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()]["batch_size"], shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    return loader
