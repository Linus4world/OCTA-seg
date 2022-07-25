from monai.transforms import Transform, MapTransform
import torch
from numpy import prod, load
from numpy.random import normal

class AddRandomNoised(MapTransform):
    """
    Generate Background signal caused by capillary vessels
    
    Parameters:
        - noise_layer_path: path to the file containing the noise_layer
        - image_weight
        - noise_weight
        - mean: mean of the gaussian distribution
        - std: standard deviation of the gaussian distribution
        - kernel_size: kernel size used for the convolution of noise
    """
    def __init__(self, *, noise_layer_path: str, image_weight=0.75, noise_weight=0.7, mean=0.3, std=0.75, kernel_size=(3,3)) -> None:
        super().__init__(keys=["image", "label"])
        if noise_layer_path is not None:
            self.noise_layer = torch.from_numpy(load(noise_layer_path)).unsqueeze(0)
        self.image_weight = image_weight
        self.noise_weight = noise_weight
        self.mean = mean
        self.std = std
        self.kernel_size = kernel_size
    
    def scale_noise_map(self, img: torch.Tensor):
        self.noise_layer = torch.nn.functional.interpolate(self.noise_layer.unsqueeze(0), size=img.shape[-2:], mode='bilinear').squeeze(0)

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.noise_layer is not None:
            img = data["image"]
            if self.noise_layer.shape != img.shape:
                self.scale_noise_map(img)
            mean = normal(loc=self.mean, scale=0.03)
            N = torch.randn_like(img) * self.std + mean
            kernel = torch.full((1,1,*self.kernel_size), 1/prod(self.kernel_size))
            N = torch.conv2d(N, kernel, padding='same')
            N = N * self.noise_layer
            img = torch.clamp(self.image_weight*img + self.noise_weight*N, 0, 1)
            data["image"] = img
        return data

class SegmentWithThresholdd(MapTransform):
    """
    Transforms a ground truth vessel map into a segmentation map.
    Each voxel with activation bigger than the given threshold is set to 1, else 0.
    """
    def __init__(self, t=0) -> None:
        super().__init__(keys=["image", "label"])
        self.t = t

    def __call__(self, data: dict[str, torch.Tensor]):
        seg = data["label"]
        mask = seg > self.t
        seg[mask] = 1
        seg[~mask] = 0
        data["label"]=seg
        return data

class ToDict(Transform):
    def __call__(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"image": data, "label": data}

class ToTorch(Transform):
    def __call__(self, data) -> torch.Tensor:
        return torch.from_numpy(data).unsqueeze(0)