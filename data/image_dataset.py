from monai.data import DataLoader, Dataset
from data.data_transforms import *
import os
from numpy import array, deg2rad
import csv
import torch
import numpy as np

from monai.data.meta_obj import set_track_meta

from utils.metrics import Task
from data.unalignedZipDataset import UnalignedZipDataset
set_track_meta(False)

def get_custom_file_paths(folder: str, name: str):
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames: list[str] = sorted(filenames)
        for filename in filenames:
            if filename.lower().endswith(name.lower()):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths

def _get_transformation(config, task: Task, phase: str, dtype=torch.float32) -> Compose:
    """
    Create and return the data transformations for 2D segmentation images the given phase.
    """
    if task == Task.VESSEL_SEGMENTATION or task == Task.GAN_VESSEL_SEGMENTATION or Task.CONSTRASTIVE_UNPAIRED_TRANSLATION:
        aug_config = config[phase.capitalize()]["data_augmentation"]
        return Compose(get_data_augmentations(aug_config, dtype))
    elif task == Task.RETINOPATHY_CLASSIFICATION:
        # TODO
        if phase == "train":
            return Compose([
                LoadImaged(keys=["image", "segmentation"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image", "segmentation"], allow_missing_keys=True),
                Rand2DElasticd(keys=["image", "segmentation"], prob=0.5, spacing=(40,40), magnitude_range=(1,4), padding_mode='zeros', allow_missing_keys=True),
                RandFlipd(keys=["image", "segmentation"], prob=.5, spatial_axis=[0, 1], allow_missing_keys=True),
                RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.5,1.5)),
                RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                RandRotate90d(keys=["image", "segmentation"], prob=.75, allow_missing_keys=True),
                RandRotated(keys=["image", "segmentation"], prob=1, range_x=deg2rad(10), padding_mode="zeros", allow_missing_keys=True),
                # RandCropOrPadd(keys=["image"], prob=1, min_factor=0.7, max_factor=1.3),
                # Resized(keys=["image", "segmentation"], shape=[1024,1024], allow_missing_keys=True),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"], enhance_vessels=config["Data"]["enhance_vessels"]),
                AddRandomErasingd(keys=["image", "segmentation"], prob=0.9, min_area=0.04, max_area=.25),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image", "segmentation"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image", "segmentation"], allow_missing_keys=True),
                Flipd(keys=["image", "segmentation"], spatial_axis=0, allow_missing_keys=True),
                Rotate90d(keys=["image", "segmentation"], k=1, allow_missing_keys=True),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"], enhance_vessels=config["Data"]["enhance_vessels"]),
                CastToTyped(keys=["image","label"], dtype=[dtype, torch.int64])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image","segmentation"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image","segmentation"], allow_missing_keys=True),
                Flipd(keys=["image","segmentation"], spatial_axis=0, allow_missing_keys=True),
                Rotate90d(keys=["image","segmentation"], k=1, allow_missing_keys=True),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"], enhance_vessels=config["Data"]["enhance_vessels"]),
                CastToTyped(keys=["image"], dtype=[dtype])
            ])
    elif task == Task.IMAGE_QUALITY_CLASSIFICATION:
        # TODO
        if phase == "train":
            return Compose([
                LoadImaged(keys=["image","segmentation"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image","segmentation"], allow_missing_keys=True),
                Rand2DElasticd(keys=["image", "segmentation"], prob=.5, spacing=(40,40), magnitude_range=(1,4), padding_mode='zeros', allow_missing_keys=True),
                # RandAdjustContrastd(keys=["image"], prob=.8, gamma=(0.5,1.5)),
                # RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                RandFlipd(keys=["image", "segmentation"], prob=.5, spatial_axis=[0, 1], allow_missing_keys=True),
                RandRotate90d(keys=["image", "segmentation"], prob=.75, allow_missing_keys=True),
                # RandRotated(keys=["image"], prob=1, range_x=deg2rad(10), padding_mode="zeros"),
                # RandCropOrPadd(keys=["image"], prob=1, min_factor=0.7, max_factor=1.3),
                # Resized(keys=["image"], shape=[1024,1024]),
                # RandBiasFieldd(keys=["image"], prob=0.1, degree=3, coeff_range=(0.1,0.3)),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"], enhance_vessels=config["Data"]["enhance_vessels"]),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image","segmentation"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image","segmentation"], allow_missing_keys=True),
                Flipd(keys=["image","segmentation"], spatial_axis=0, allow_missing_keys=True),
                Rotate90d(keys=["image","segmentation"], k=1, allow_missing_keys=True),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"], enhance_vessels=config["Data"]["enhance_vessels"]),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        else:
            return Compose([
               LoadImaged(keys=["image","segmentation"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image","segmentation"], allow_missing_keys=True),
                Flipd(keys=["image","segmentation"], spatial_axis=0, allow_missing_keys=True),
                Rotate90d(keys=["image","segmentation"], k=1, allow_missing_keys=True),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"], enhance_vessels=config["Data"]["enhance_vessels"]),
                CastToTyped(keys=["image"], dtype=[dtype])
            ])
    elif task == Task.AREA_SEGMENTATION:
        # TODO
        if phase == "train":
            return Compose([
                LoadImaged(keys=["image","segmentation", "label"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation", "label"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image","segmentation"], allow_missing_keys=True),
                Rand2DElasticd(keys=["image","segmentation", "label"], prob=.8, spacing=(40,40), magnitude_range=(1,4), padding_mode='zeros', allow_missing_keys=True),
                RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.5,1.5)),
                RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.25), sigma_y=(0.25, 1.5)),
                RandFlipd(keys=["image", "segmentation", "label"], prob=.5, spatial_axis=[0, 1], allow_missing_keys=True),
                RandRotate90d(keys=["image", "segmentation", "label"], prob=.75, allow_missing_keys=True),
                RandRotated(keys=["image", "segmentation", "label"], prob=1, range_x=deg2rad(10), padding_mode="zeros", allow_missing_keys=True),
                # AddRandomErasingd(keys=["image", "segmentation"], prob=0.9, min_area=0.04, max_area=.25),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AsDiscreted(keys=["label"], threshold=.5),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"], enhance_vessels=config["Data"]["enhance_vessels"]),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image","segmentation", "label"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation", "label"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image","segmentation"], allow_missing_keys=True),
                Flipd(keys=["image","segmentation", "label"], spatial_axis=0, allow_missing_keys=True),
                Rotate90d(keys=["image","segmentation", "label"], k=1, allow_missing_keys=True),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"]),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])
        else:
            return Compose([
                LoadImaged(keys=["image","segmentation", "label"], image_only=True, allow_missing_keys=True),
                ScaleIntensityd(keys=["image", "segmentation", "label"], minv=0, maxv=1, allow_missing_keys=True),
                AddChanneld(keys=["image","segmentation"], allow_missing_keys=True),
                Rotate90d(keys=["image","segmentation", "label"], k=1, allow_missing_keys=True),
                Flipd(keys=["image","segmentation", "label"], spatial_axis=0, allow_missing_keys=True),
                FuseImageSegmentationd(image_key_label="image", seg_key_label="segmentation", target_label="image", use_diff=config["Data"]["use_background"]),
                CastToTyped(keys=["image", "label"], dtype=[dtype, torch.int64])
            ])

def get_post_transformation(config: dict, phase: str, task: Task, num_classes=2) -> tuple[Compose]:
    """
    Create and return the data transformation that is applied to the model prediction before inference.
    """
    aug_config: dict = config[phase.capitalize()]["post_processing"]
    return Compose(get_data_augmentations(aug_config.get("prediction"))), Compose(get_data_augmentations(aug_config.get("label")))
    # if task == Task.VESSEL_SEGMENTATION:
    #     return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), RemoveSmallObjects(396)]), Compose([CastToType(dtype=torch.uint8)])
    # elif task == Task.GAN_VESSEL_SEGMENTATION:
    #     if phase != "test" or config["Test"]["inference"] == "S":
    #         return Compose([AsDiscrete(threshold=0.5), RemoveSmallObjects(256)]), Compose()
    #     else:
    #         return Compose(), Compose()
    # elif task == Task.CONSTRASTIVE_UNPAIRED_TRANSLATION:
    #      return Compose(), Compose()
    # elif task == task == Task.AREA_SEGMENTATION:
    #     return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]), Compose([CastToType(dtype=torch.uint8)])
    # elif num_classes>1:
    #     return Compose([Activations(softmax=True)]), Compose([AsDiscrete(to_onehot=num_classes)])
    # else:
    #     return Compose([AsOneHot(num_classes=3)]), Compose([AsDiscrete(to_onehot=3)])


def get_dataset(config: dict, phase: str, batch_size=None) -> DataLoader:
    """
    Creates and return the dataloader for the given phase.
    """
    task = config["General"]["task"]
    transform = _get_transformation(config, task, phase, dtype=torch.float16 if config["General"]["amp"] else torch.float32)

    data_settings: dict = config[phase.capitalize()]["data"]
    data = dict()
    for key, val in data_settings.items():
        paths  = get_custom_file_paths(val["folder"], val["suffix"])
        paths.sort()
        if "split" in val:
            with open(val["split"], 'r') as f:
                lines = f.readlines()
                indices = [int(line.rstrip()) for line in lines]
                paths = array(paths)[indices].tolist()
        data[key] = paths
        data[key+"_path"] = paths

    if task == Task.VESSEL_SEGMENTATION:
        max_length = max([len(l) for l in data.values()])
        for k,v in data.items():
            data[k] = np.resize(np.array(v), max_length).tolist()
        train_files = [dict(zip(data, t)) for t in zip(*data.values())]
    elif task == Task.GAN_VESSEL_SEGMENTATION:
        if phase == "validation":
            max_length = max([len(l) for l in data.values()])
            for k,v in data.items():
                data[k] = np.resize(np.array(v), max_length).tolist()
            train_files = [dict(zip(data, t)) for t in zip(*data.values())]
        else:
            data_set = UnalignedZipDataset(data, transform, phase, config["General"]["inference"])
            loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()]["batch_size"], shuffle=phase!="test", num_workers=8, pin_memory=torch.cuda.is_available())
            return loader
    elif task == Task.CONSTRASTIVE_UNPAIRED_TRANSLATION:
        # TODO
        if phase != "test":
            A_paths = get_custom_file_paths(*config[phase.capitalize()]["synthetic_images"])
        else:
            A_paths = image_paths
            image_paths = None
        data_set = UnalignedZipDataset(A_paths, image_paths, None, transform, phase, config["General"]["inference"])
        loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()]["batch_size"], shuffle=phase!="test", num_workers=8, pin_memory=torch.cuda.is_available())
        return loader
    elif task == Task.RETINOPATHY_CLASSIFICATION or task == Task.IMAGE_QUALITY_CLASSIFICATION:
        # TODO
        if config["Data"]["use_segmentation"] or config["Data"]["enhance_vessels"]:
            seg_paths = config["Test"]["dataset_segmentations"] if phase == "test" else config["Data"]["dataset_segmentations"]
            seg_paths = get_custom_file_paths(*seg_paths)
            seg_paths = array(seg_paths)[indices].tolist()
        
        if config["Data"]["dataset_labels"] != '':
            reader = csv.reader(open(config["Data"]["dataset_labels"], 'r'))
            next(reader)
            labels = [int(v) for k, v in reader]
            labels = list(array(labels)[indices])
            if config["Data"]["use_segmentation"] or config["Data"]["enhance_vessels"]:
                train_files = [{"image": img, "segmentation": seg, "path": img, "label": torch.tensor(label)} for img, seg, label in zip(image_paths, seg_paths, labels)]
            else:
                train_files = [{"image": img, "path": img, "label": torch.tensor(label)} for img, label in zip(image_paths, labels)]
        else:
            train_files = [{"image": img, "path": img} for img in image_paths]
    elif task == Task.AREA_SEGMENTATION:
        # TODO
        if config["Data"]["use_segmentation"] or config["Data"]["enhance_vessels"]:
            seg_paths = config["Test"]["dataset_segmentations"] if phase == "test" else config["Data"]["dataset_segmentations"]
            seg_paths = get_custom_file_paths(*seg_paths)
            seg_paths = array(seg_paths)[indices].tolist()
        
        placeholder_path = os.path.join("/".join(config[phase.capitalize()]["dataset_path"].split('/')[:-1]),'placeholder.png')
        if config["Data"]["dataset_labels"]:
            train_files = []
            for i in range(len(image_paths)):
                img_path = image_paths[i]
                name = img_path.split('/')[-1]
                labels = []
                for path in config["Data"]["dataset_labels"]:
                    label_path = os.path.join(path, name)
                    if os.path.exists(label_path):
                        labels.append(label_path)
                    else:
                        labels.append(placeholder_path)
                if config["Data"]["use_segmentation"]:
                    train_files.append({"image": img_path, "segmentation": seg_paths[i], "path": img_path, "label": labels})
                else:
                    train_files.append({"image": img_path, "path": img_path, "label": labels})
        else:
            train_files = [{"image": img, "path": img} for img in image_paths]


    data_set = Dataset(train_files, transform=transform)
    loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()]["batch_size"], shuffle=phase!="test", num_workers=8, pin_memory=torch.cuda.is_available())
    return loader
