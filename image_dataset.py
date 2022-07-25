from monai.data import DataLoader, Dataset
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, ScaleIntensity, RandFlipd
from data_transforms import AddRandomNoised, SegmentWithThresholdd, ToDict, ToTorch
import os

def get_custom_file_paths(folder, name):
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            if filename.endswith(name):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths

def get_transformations(config) -> tuple:
    r = AddRandomNoised(noise_layer_path=config["Data"]["noise_map_path"])
    train_transform = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(0, 1),
        ToTorch(),
        ToDict(),
        r,
        SegmentWithThresholdd(0),
        # RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=0),
        # RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=1)
    ])
    val_transform = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(0, 1),
        ToTorch(),
        ToDict(),
        r,
        SegmentWithThresholdd(0),
    ])
    
    return train_transform, val_transform

def get_post_trasnformation():
    return Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def get_datasests(config) -> tuple[DataLoader]:
    r = AddRandomNoised(noise_layer_path=config["Data"]["noise_map_path"])
    train_transform, val_transform = get_transformations(config)

    train_set = Dataset(get_custom_file_paths(f'{config["Data"]["dataset_path"]}train/', '.png'), transform=train_transform)
    val_set = Dataset(get_custom_file_paths(f'{config["Data"]["dataset_path"]}val/', '.png'), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=config["Train"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config["Train"]["batch_size"], shuffle=False, num_workers=4)
    return train_loader, val_loader
