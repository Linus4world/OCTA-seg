from monai.data import DataLoader, Dataset
from monai.transforms import *
from data_transforms import *
import os
from numpy import array, deg2rad
import csv
import torch

from monai.data.meta_obj import set_track_meta

from utils.metrics import Task
from utils.unalignedZipDataset import UnalignedZipDataset
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
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Rotate90d(keys=["image"], k=1),
                Flipd(keys=["image"], spatial_axis=0),
                SplitImageLabeld(keys=["image", "label"]),
                AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0,1]),
                RandCropOrPadd(keys=["image", "label"], prob=0.5, min_factor=0.8, max_factor=1.2),
                RandRotate90d(keys=["image", "label"], prob=.75),
                RandRotated(keys=["image", "label"], range_x=deg2rad(10)),
                Rand2DElasticd(keys=["image", "label"], prob=.5, spacing=(20,20), magnitude_range=(2,4), padding_mode='zeros'),
                Resized(keys=["image", "label"], shape=(304,304)),
                # Resized(keys=["image", "label"], shape=(1024,1024)),
                # RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5,2)),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddLineArtifact(keys=["image"]),
                AsDiscreted(keys=["label"], threshold=0.001),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        elif phase == "validation":
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Rotate90d(keys=["image"], k=1),
                Flipd(keys=["image"], spatial_axis=0),
                SplitImageLabeld(keys=["image", "label"]),
                AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
                Resized(keys=["image", "label"], shape=(304,304)),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AsDiscreted(keys=["label"],threshold=0.001),
                CastToTyped(keys=["image", "label"], dtype=dtype)
            ])
        else:
            return Compose([
                LoadImaged(keys=["image"], image_only=True),
                ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                AddChanneld(keys=["image"]),
                Rotate90d(keys=["image"], k=1),
                Flipd(keys=["image"], spatial_axis=0),
                CastToTyped(keys=["image"], dtype=dtype)
            ])
    elif task == Task.GAN_VESSEL_SEGMENTATION:
        if phase == "train":
            return Compose([
                LoadImaged(keys=["real_A", "real_B"], image_only=True),
                ScaleIntensityd(keys=["real_A", "real_B"], minv=0, maxv=1),
                AddChanneld(keys=["real_A", "real_B"]),
                RandFlipd(keys=["real_A", "real_B"], prob=0.5, spatial_axis=[0,1]),
                # Rand2DElasticd(keys=["real_A", "real_B"], prob=.5, spacing=(40,40), magnitude_range=(1,4), padding_mode='zeros'),
                RandRotate90d(keys=["real_A", "real_B"], prob=.75),
                RandRotated(keys=["real_A", "real_B"], prob=1, range_x=deg2rad(10), padding_mode="zeros"),
                Resized(keys=["real_B"], shape=(1216,1216)),
                RandCropOrPadd(keys=["real_A", "real_B"], prob=1, min_factor=0.25, max_factor=0.25),
                ScaleIntensityd(keys=["real_A", "real_B"], minv=0, maxv=1),
                SplitImageLabeld(keys=["real_A", "real_A_seg"]),
                AsDiscreted(keys=["real_A_seg"],threshold=0.1),
                CastToTyped(keys=["real_A", "real_A_seg", "real_B"], dtype=dtype)
            ])
    elif task == Task.RETINOPATHY_CLASSIFICATION:
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

def get_post_transformation(task: Task, num_classes=2) -> tuple[Compose]:
    """
    Create and return the data transformation that is applied to the model prediction before inference.
    """
    if task == Task.VESSEL_SEGMENTATION:
        return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5), KeepLargestConnectedComponent()]), Compose([CastToType(dtype=torch.uint8)])
    elif task == Task.GAN_VESSEL_SEGMENTATION:
        return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]), Compose()
    elif task == task == Task.AREA_SEGMENTATION:
        return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)]), Compose([CastToType(dtype=torch.uint8)])
    elif num_classes>1:
        return Compose([Activations(softmax=True)]), Compose([AsDiscrete(to_onehot=num_classes)])
    else:
        return Compose([AsOneHot(num_classes=3)]), Compose([AsDiscrete(to_onehot=3)])


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
        if phase != "test" and "dataset_labels" in config["Data"]:
            label_paths = get_custom_file_paths(*config["Data"]["dataset_labels"])
            label_paths = array(label_paths)[indices].tolist()
            train_files = [{"image": path, "label":l_path, "path": path} for path, l_path in zip(image_paths, label_paths)]
        else:
            train_files = [{"image": path, "path": path} for path in image_paths]
    elif task == Task.GAN_VESSEL_SEGMENTATION:
        if phase != "test":
            A_paths = get_custom_file_paths(*config["Data"]["synthetic_images"])
        else:
            A_paths = None
        data_set = UnalignedZipDataset(A_paths, image_paths, transform)
        loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()]["batch_size"], shuffle=phase!="test", num_workers=8, pin_memory=torch.cuda.is_available())
        return loader
    elif task == Task.RETINOPATHY_CLASSIFICATION or task == Task.IMAGE_QUALITY_CLASSIFICATION:
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
