from monai.data import DataLoader, Dataset
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, ScaleIntensity, RandFlipd, Rotate, Flip, RandScaleCropd, RandRotated, AsDiscreted, AddChannel
from data_transforms import AddRealNoised, ToDict, Resized, AddLineArtifact
import os
from numpy import deg2rad

from monai.data.meta_obj import set_track_meta
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

def _get_transformation(config, phase: str) -> Compose:
    """
    Create and return the data transformations for 2D segmentation images the given phase.
    """
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
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            RandScaleCropd(keys=["image", "label"], roi_scale=0.5, max_roi_scale=1, random_center=True, random_size=True),
            RandRotated(keys=["image", "label"], range_x=10, range_y=10),
            Resized(keys=["image", "label"], shape=(1024,1024)),
            AddLineArtifact(keys=["image"]),
            AsDiscreted(keys=["label"],threshold=0.1)
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
            AsDiscreted(keys=["label"],threshold=0.1)
        ])
    else:
        return Compose([
            LoadImage(image_only=True),
            ScaleIntensity(0, 1),
            AddChannel(),
            Rotate(deg2rad(270)),
            Flip(0)
        ])

def get_post_transformation():
    """
    Create and return the data transformation that is applied to the model prediction before inference.
    """
    return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def get_dataset(config: dict, phase: str) -> DataLoader:
    """
    Creates and return the dataloader for the given phase.
    """
    transform = _get_transformation(config, phase)
    train_set = Dataset(get_custom_file_paths(config[phase.capitalize()]["dataset_path"], '.png'), transform=transform)
    loader = DataLoader(train_set, batch_size=config[phase.capitalize()]["batch_size"], shuffle=phase=="train", num_workers=4)
    return loader
