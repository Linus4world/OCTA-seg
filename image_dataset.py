from monai.data import DataLoader, Dataset
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, ScaleIntensity, RandFlipd, Rotate, Flip, RandScaleCropd, RandRotated
from data_transforms import AddRealNoised, SegmentWithThresholdd, ToDict, ToTorch, Resized
import os
from numpy import deg2rad

def get_custom_file_paths(folder, name):
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            if filename.endswith(name):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths

def get_transformation(config, phase: str) -> tuple:
    if phase == "train":
        return Compose([
            LoadImage(image_only=True),
            ScaleIntensity(0, 1),
            ToTorch(),
            Rotate(deg2rad(270)),
            Flip(0),
            ToDict(),
            # AddRandomNoised(noise_layer_path=config["Data"]["noise_map_path"]),
            AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
            RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=1),
            RandScaleCropd(keys=["image", "label"], roi_scale=0.9, max_roi_scale=1, random_center=True, random_size=True),
            RandRotated(keys=["image", "label"], range_x=10, range_y=10),
            Resized(keys=["image", "label"], shape=(1024,1024)),
            SegmentWithThresholdd(keys=["label"], t=0),
        ])
    elif phase == "validation":
        return Compose([
            LoadImage(image_only=True),
            ScaleIntensity(0, 1),
            ToTorch(),
            Rotate(deg2rad(270)),
            Flip(0),
            ToDict(),
            AddRealNoised(keys=["image"], noise_paths=get_custom_file_paths(config["Data"]["real_noise_path"], "art_ven_gray_z.png"), noise_layer_path=config["Data"]["noise_map_path"]),
            SegmentWithThresholdd(keys=["label"], t=0),
        ])
    else:
        return Compose([
            LoadImage(image_only=True),
            ScaleIntensity(0, 1),
            ToTorch(),
            Rotate(deg2rad(270)),
            Flip(0)
        ])

def get_post_transformation():
    return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def get_dataset(config: dict, phase: str) -> DataLoader:
    transform = get_transformation(config, phase)
    train_set = Dataset(get_custom_file_paths(config[phase.capitalize()]["dataset_path"], '.png'), transform=transform)
    loader = DataLoader(train_set, batch_size=config[phase.capitalize()]["batch_size"], shuffle=phase=="train", num_workers=4)
    return loader
