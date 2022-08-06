from monai.transforms import Transform, MapTransform
import torch
from numpy import prod, load, array
from numpy.random import normal
import math
import random
from PIL import Image

class AddRealNoised(MapTransform):
    def __init__(self, keys: list[str], noise_paths: list[str], noise_layer_path: str) -> None:
        super().__init__(keys=keys)
        self.noise_paths = noise_paths
        self.noise_layer = torch.from_numpy(load(noise_layer_path)).to(dtype=torch.float32).unsqueeze(0)

    def get_real_noise(self, k=3):
        t = torch.zeros(1)
        for n in random.choices(self.noise_paths, k=k):
            t = torch.maximum(t, torch.from_numpy(array(Image.open(n))).to(dtype=torch.float32))
        t = t.unsqueeze(0)/255
        return t

    def scale_noise_map(self, noise_layer: torch.Tensor, shape):
        return torch.nn.functional.interpolate(noise_layer.unsqueeze(0), size=shape, mode='bicubic').squeeze(0)

    def blur(self, img: torch.Tensor, kernel_size=(3,3)):
        kernel = torch.full((1,1,*kernel_size), 1/math.prod(kernel_size))
        blurred_img = torch.conv2d(img.unsqueeze(0), kernel, padding='same').squeeze(0)
        blurred_img = blurred_img - blurred_img.min()
        blurred_img = (blurred_img / blurred_img.max())
        return blurred_img

    def get_gaussian_noise(self, shape, mean=0, std=1):
        return torch.randn(shape)*std+mean

    def get_multi_level_noise(self, shape, l=100, kernel_size=[3,3]):
        s_0 = torch.tensor(shape)
        s = (1, *(s_0[-2:] / l).int().tolist())
        N = torch.randn(s)*0.8+2
        N = torch.clamp(N, 0, 3)
        N = torch.nn.functional.interpolate(N.unsqueeze(0), size=shape[-2:], mode='bicubic').squeeze(0)
        kernel = torch.full((1,1,*kernel_size), 1/math.prod(kernel_size))
        N = torch.conv2d(N, kernel, padding='same')
        return N

    def add_noise(self, img: torch.Tensor, lambda_img=0.9, lambda_N_real=0.9, lambda_N_gauss=1.2):
        if self.noise_layer is not None:
            if self.noise_layer.shape != img.shape:
                self.noise_layer = self.scale_noise_map(self.noise_layer, img.shape[-2:])

        N_real = self.get_real_noise()
        if random.uniform(0,1)>0.5:
            N_real = self.blur(N_real, kernel_size=[3,3])*0.55

        N_gauss = self.get_gaussian_noise(img.shape, 0, 1)
        N_gauss = self.blur(N_gauss, (3,3)) * self.noise_layer
        N_gauss_center = random.uniform(-0.45,-0.7) #-0.5
        N_gauss = N_gauss + N_gauss_center

        multi_level_noise = self.get_multi_level_noise(img.shape)

        img = lambda_img * img

        img = self.blur(img, (3,3)) * self.noise_layer
        img = torch.maximum(img, N_real * lambda_N_real * multi_level_noise)

        img = img + N_gauss*lambda_N_gauss

        return torch.clamp(img, 0, 1)

    def __call__(self, data: torch.Tensor):
        for key in self.keys:
            img = data[key]
            img = torch.clamp(img*3,0,1)
            img = self.add_noise(img)
            data[key] = img
        return data

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
    def __init__(self, *, noise_layer_path: str, image_weight=0.65, noise_weight=0.7, mean=0.4, std=0.25, kernel_size=(3,3)) -> None:
        super().__init__(keys=["image", "label"])
        if noise_layer_path is not None:
            self.noise_layer = torch.from_numpy(load(noise_layer_path)).to(dtype=torch.float32).unsqueeze(0)
        self.image_weight = image_weight
        self.noise_weight = noise_weight
        self.mean = mean
        self.std = std
        self.kernel_size = kernel_size
    
    def scale_noise_map(self, img: torch.Tensor):
        self.noise_layer = torch.nn.functional.interpolate(self.noise_layer.unsqueeze(0), size=img.shape[-2:], mode='bilinear').squeeze(0)

    def get_multi_level_noise(self, shape, levels=(128, 1), kernel_size=(3,3)) -> torch.Tensor:
        s_0 = torch.tensor(shape)
        N = 1
        for l in levels:
            s = (1, *(s_0[-2:] / l).int().tolist())
            N_l = torch.randn(s)
            N_l = torch.nn.functional.interpolate(N_l.unsqueeze(0), size=shape[-2:], mode='bilinear').squeeze(0)
            if l == 1:
                N_l *= 5
            kernel = torch.full((1,1,*kernel_size), 1/prod(kernel_size))
            N_l = torch.conv2d(N_l, kernel, padding='same')
            N = N+N_l
        N = N/len(levels)
        N = N-N.mean()
        N = N/N.std()
        return N

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.noise_layer is not None:
            img = data["image"]
            if self.noise_layer.shape != img.shape:
                self.scale_noise_map(img)
            mean = normal(loc=self.mean, scale=0.03)
            N = self.get_multi_level_noise(img.shape,(128,1))
            N = N * self.std + mean
            N = N * self.noise_layer
            img = self.image_weight*img + self.noise_weight*N
            kernel = torch.full((1,1,*self.kernel_size), 1/prod(self.kernel_size))
            img = torch.conv2d(img, kernel, padding='same')
            img = torch.clamp(img, 0, 1)
            data["image"] = img
        return data

class SegmentWithThresholdd(MapTransform):
    """
    Transforms a ground truth vessel map into a segmentation map.
    Each voxel with activation bigger than the given threshold is set to 1, else 0.
    """
    def __init__(self, keys: list[str], t=0) -> None:
        super().__init__(keys=keys)
        self.t = t

    def __call__(self, data: dict[str, torch.Tensor]):
        for key in self.keys:
            seg = data[key]
            mask = seg > self.t
            seg[mask] = 1
            seg[~mask] = 0
            data[key]=seg
        return data

class ToDict(Transform):
    def __call__(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"image": data, "label": torch.clone(data)}

class ToTensor(Transform):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.unsqueeze(0)

class Resized(MapTransform):
    def __init__(self, keys: list[str], shape) -> None:
        super().__init__(keys=keys)
        self.shape = shape

    def __call__(self, data) -> torch.Tensor:
        for key in self.keys:
            d = data[key]
            data[key] = torch.nn.functional.interpolate(d.unsqueeze(0), size=self.shape, mode='bilinear').squeeze(0)
        return data