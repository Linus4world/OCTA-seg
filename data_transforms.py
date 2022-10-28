import torch
from numpy import prod, load, array
from numpy.random import normal
import math
import random
from PIL import Image

from monai.networks.nets import DynUNet
from utils.metrics import Task
from monai.transforms import *

class RandomDecreaseResolutiond(MapTransform):
    def __init__(self, keys: tuple[str], p=1, max_factor=0.25) -> None:
        super().__init__(keys, True)
        self.max_factor = max_factor
        self.p = p

    def __call__(self, data):
        if random.uniform(0,1)<self.p:
            for key in self.keys:
                d = data[key]
                size = d.shape
                factor = random.uniform(self.max_factor,1)
                d = torch.nn.functional.interpolate(d.unsqueeze(0), scale_factor=factor)
                d = torch.nn.functional.interpolate(d, size=size[1:]).squeeze(0)
                data[key]=d
        return data

class AddRandomGaussianNoiseChanneld(MapTransform):
    def __init__(self, keys:  tuple[str]) -> None:
        super().__init__(keys, True)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            noise = torch.sigmoid(torch.rand_like(img))
            img = torch.cat((img, noise), dim=0)
            data[key] = img
        return data

class AsOneHot():
    def __init__(self, num_classes=3) -> None:
        self.num_classes = num_classes

    def __call__(self, input: torch.Tensor):
        ret = []
        if input<0:
            return torch.tensor([1, *[0]*(self.num_classes-1)])
        if input>self.num_classes-1:
            return torch.tensor([*[0]*(self.num_classes-1), 1])
        for c in range(self.num_classes):
            if c-1<input<c+1:
                ret.append(1-abs(c-input))
            else:
                ret.append(0)
        return torch.tensor(ret)

class FuseImageSegmentationd():
    def __init__(self, image_key_label: str, seg_key_label: str, target_label: str, use_diff=False, enhance_vessels=False):
        self.use_diff=use_diff
        self.target_label=target_label
        self.image_key_label = image_key_label
        self.seg_key_label=seg_key_label
        self.enhance_vessels = enhance_vessels

    def __call__(self, data: dict):
        if self.seg_key_label not in data:
            return data
        img: torch.Tensor = data[self.image_key_label]
        seg: torch.Tensor = data[self.seg_key_label]
        del data[self.image_key_label]
        del data[self.seg_key_label]
        if self.use_diff:
            diff=img.clone()
            diff[seg>0]=0
            fused = torch.cat([img,seg,diff])
        elif self.enhance_vessels:
            fused = img.clone()
            fused[seg>0.5]=1
        else:
            fused = torch.cat([img,seg])
        data[self.target_label]=fused
        return data

class VesselSegmentationd(MapTransform):
    """
    NOT USED
    """
    def __init__(self, keys: tuple[str], target_key: str, config: dict, get_post_transformation) -> None:
        assert len(keys)==1
        self.USE_SEG_INPUT = config["Data"]["use_segmentation"]
        model_path: str = config["Train"]["model_path"]
        num_layers = config["General"]["num_layers"]
        kernel_size = config["General"]["kernel_size"]
        self.device = "cpu"#torch.device(config["General"]["device"])
        self.target_key = target_key
        super().__init__(keys, False)

        if self.USE_SEG_INPUT:
            self.segmentation_model = DynUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=(3, *[kernel_size]*num_layers,3),
                strides=(1,*[2]*num_layers,1),
                upsample_kernel_size=(1,*[2]*num_layers,1),
            ).to(self.device)
            checkpoint = torch.load(model_path)
            self.segmentation_model.load_state_dict(checkpoint['model'])
            self.segmentation_model.eval()
            self.post_itermediate, _ = get_post_transformation(Task.VESSEL_SEGMENTATION)

    def __call__(self, data: dict):
        if self.USE_SEG_INPUT:
            img: torch.Tensor = data[self.keys[0]]

            with torch.no_grad():
                seg = self.segmentation_model(img.to(device=self.device).unsqueeze(0))
                seg = [self.post_itermediate(inter) for inter in seg][0].cpu()
            data[self.target_key] = seg
        return data

class AddRandomErasingd(MapTransform):
    def __init__(self, keys: tuple[str], prob = 1, min_area=0.04, max_area = 0.25):
        super().__init__(keys, True)
        self.prob = prob
        self.max_area = max_area
        self.min_area = min_area
       
    def __call__(self, data: dict):

        if random.uniform(0,1)>self.prob:
            return data

        EPSISON = 1e-8
        h = random.uniform(EPSISON,1)
        w = random.uniform(EPSISON,1)
        area = random.uniform(self.min_area, self.max_area)
        mult = math.sqrt(area / (h*w))
        h = min(mult*h, 1)
        w = min(mult*w,1)
        s_h = None

        for key in self.keys:
            if key in data:
                img: torch.Tensor = data[key]
                if s_h is None:
                    h_i = max(1,math.floor(h * img.shape[1]))
                    w_i = max(1,math.floor(w * img.shape[2]))
                    rect = torch.normal(0.5,0.1,([img.shape[0],h_i,w_i]))
                    s_h = random.randint(0,img.shape[1]-h_i)
                    s_w = random.randint(0,img.shape[2]-w_i)

                img[:,s_h:s_h+h_i,s_w:s_w+w_i] = rect
                
                data[key] = img
        return data

class AddLineArtifact(MapTransform):
    """
    Generates a blurry horizontal line with is a common image artifact in OCTA images 
    """
    def __init__(self, keys: tuple[str]) -> None:
        """
        Generates a blurry horizontal line with is a common image artifact in OCTA images
        
        Parameters:
            - keys: List of dict keys where the artifact should be applied to
        """
        super().__init__(keys, False)
        self.c = torch.tensor([[0.0250, 0.0750, 0.3750, 0.8750, 1.0000, 0.8750, 0.3750, 0.0750, 0.0250]]).unsqueeze(-1)

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            start = random.randint(0,img.shape[-2]-9)
            s = slice(start,start+9)
            line = img[:,s,:].unsqueeze(0)
            line = torch.conv2d(line, weight=torch.full((1,1,7,7), 1/50), padding="same")
            img[:,s,:] = img[:,s,:]*(1-self.c) + self.c * line[0,:,:,:]
            data[key] = img
        return data

class AddRealNoised(MapTransform):
    """
    Generate Background signal caused by capillary vessels using simulated deep vascular complex (DVC) maps
    """
    def __init__(self, keys: list[str], noise_paths: list[str], noise_layer_path: str) -> None:
        """
        Generate Background signal caused by capillary vessels using simulated deep vascular complex (DVC) maps
        
        Parameters:
            - noise_paths: directory of the DVC maps
            - noise_layer_path: path to the file containing the noise_layer map
            - keys: List of dict keys where the noise should be applied to
        """
        super().__init__(keys=keys)
        self.noise_paths = noise_paths
        self.noise_layer = torch.from_numpy(load(noise_layer_path)).to(dtype=torch.float32).unsqueeze(0)

    def _get_real_noise(self, k=3):
        """
        Loads k DVC vessel maps and combines them via MIP.
        """
        t = torch.zeros(1)
        for n in random.choices(self.noise_paths, k=k):
            t = torch.maximum(t, torch.from_numpy(array(Image.open(n))).to(dtype=torch.float32))
        t = t.unsqueeze(0)/255
        return t

    def _scale_noise_map(self, noise_layer: torch.Tensor, shape):
        """
        Scales the given noise map to the given shape via bicubic interpolation.
        """
        return torch.nn.functional.interpolate(noise_layer.unsqueeze(0), size=shape, mode='bicubic').squeeze(0)

    def _blur(self, img: torch.Tensor, kernel_size=(3,3)):
        """
        Blurs the given noise tensor with an averaging kernel of the given size and normalizes it to mean=0 and std=1.
        """
        kernel = torch.full((1,1,*kernel_size), 1/math.prod(kernel_size))
        blurred_img = torch.conv2d(img.unsqueeze(0), kernel, padding='same').squeeze(0)
        blurred_img = blurred_img - blurred_img.min()
        blurred_img = (blurred_img / blurred_img.max())
        return blurred_img

    def _get_gaussian_noise(self, shape, mean=0, std=1):
        """
        Generates a tensor of the given shape with gaussian white noise
        """
        return torch.randn(shape)*std+mean

    def _get_multi_level_noise(self, shape, l=100, kernel_size=[3,3]):
        """
        Generates a tensor of the given shape divided by l with gaussian white noise, and then resizes it to the target shape with bicubic interpolation.
        The noise is then smoothened by an averaging kernel of the given kernel_size.
        This creates not a pixel, but area based noise map.
        """
        s_0 = torch.tensor(shape)
        s = (1, *(s_0[-2:] / l).int().tolist())
        N = torch.randn(s)*0.8+2
        N = torch.clamp(N, 0, 3)
        N = torch.nn.functional.interpolate(N.unsqueeze(0), size=shape[-2:], mode='bicubic').squeeze(0)
        kernel = torch.full((1,1,*kernel_size), 1/math.prod(kernel_size))
        N = torch.conv2d(N, kernel, padding='same')
        return N

    def add_noise(self, img: torch.Tensor, lambda_img=1, lambda_N_real=0.9, lambda_N_gauss=1.2):
        """
        Adds noise to the given image tensor by combining gaussian white noise, real DVC noise maps and bias fields.
        The final image output is given by:

            img_blur = blur(λ_img*img) * noise_layer \n
            N_gauss = blur(N(μ, 1)), with μ~U(-0.45, -0.7) \n
            img = MAX[img_blur, λ_real * N_real * multi_level_noise] + λ_gauss * N_gauss

        where N_real can either be blurred or not.
        """
        if self.noise_layer is not None:
            if self.noise_layer.shape != img.shape:
                self.noise_layer = self._scale_noise_map(self.noise_layer, img.shape[-2:])

        N_real = self._get_real_noise()
        if random.uniform(0,1)>0.5:
            N_real = self._blur(N_real, kernel_size=[3,3])*0.55

        N_gauss = self._get_gaussian_noise(img.shape, 0, 1)
        N_gauss = self._blur(N_gauss, (3,3)) * self.noise_layer
        N_gauss_center = random.uniform(-0.45,-0.7) #-0.5
        N_gauss = N_gauss + N_gauss_center

        multi_level_noise = self._get_multi_level_noise(img.shape)

        img = self._blur(img, (3,3)) * self.noise_layer
        img = random.uniform(lambda_img*0.8,lambda_img*1.1) * img
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
    @Depricated Use the newer AddRealNoised\n
    Generate Background signal caused by capillary vessels
    """
    def __init__(self, *, noise_layer_path: str, image_weight=0.65, noise_weight=0.7, mean=0.4, std=0.25, kernel_size=(3,3)) -> None:
        """
        @Depricated Use the newer AddRealNoised\n
        Generate Background signal caused by capillary vessels
        
        Parameters:
            - noise_layer_path: path to the file containing the noise_layer
            - image_weight
            - noise_weight
            - mean: mean of the gaussian distribution
            - std: standard deviation of the gaussian distribution
            - kernel_size: kernel size used for the convolution of noise
        """
        super().__init__(keys=["image", "label"])
        from warnings import warn
        warn("This function is depricated! Use the newer AddRealNoised")
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

class SplitImageLabel(Transform):
    """
    Clones the image tensor to create a label tensr and puts them in a dictionary.
    """
    def __call__(self, data: torch.Tensor, keys=["image", "label"]) -> dict[str, torch.Tensor]:
        return {keys[0]: data, keys[1]: torch.clone(data)}

class SplitImageLabeld(MapTransform):
    """
    Clones the image tensor to create a label tensr and puts them in a dictionary.
    """
    def __call__(self, data: dict) -> dict[str, torch.Tensor]:
        data[self.keys[1]] = torch.clone(data[self.keys[0]])
        return data

class ToTensor(Transform):
    """
    Adds an additional channel dimension to the data tensor
    """
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.unsqueeze(0)

class Resized(MapTransform):
    """
    Resize image to the given shape using bilinear interpolation
    """
    def __init__(self, keys: list[str], shape: tuple[int]) -> None:
        super().__init__(keys=keys)
        self.shape = shape

    def __call__(self, data) -> torch.Tensor:
        for key in self.keys:
            if key in data:
                d = data[key]
                data[key] = torch.nn.functional.interpolate(d.unsqueeze(0), size=self.shape, mode='bilinear').squeeze(0)
        return data

class RandCropOrPadd(MapTransform):
    """
    Randomly crop or pad the image with a random zoom factor.
    """
    def __init__(self, keys: list[str], prob=0.1, min_factor=1, max_factor=1) -> None:
        """
        Randomly crop or pad the image with a random zoom factor.
        If zoom_factor > 1, the image will be zero-padded to fit the larger image shape.
        If zoom_factor > 1, the image will be cropped at a random center to fit the larger image shape.

        Parameters:
            - keys: List of dict keys where the noise should be applied to
            - prob: Probability with which the transform is applied
            - min_factor: Smallest allowed zoom factor
            - max_factor: Largest allowed zoom factor
        """
        super().__init__(keys)
        self.prob = prob
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, data):
        if random.uniform(0,1)<self.prob:
            factor = random.uniform(self.min_factor, self.max_factor)
            slice_x = slice_y = None
            for k in self.keys:
                d: torch.Tensor = data[k]
                if factor<1:
                    if slice_x is None:
                        s_x = int(d.shape[1]*factor)
                        s_y = int(d.shape[2]*factor)
                        start_x = random.randint(0, d.shape[1]-s_x)
                        start_y = random.randint(0, d.shape[2]-s_y)
                        slice_x = slice(start_x, start_x + s_x)
                        slice_y = slice(start_y, start_y + s_y)
                    d = d[:,slice_x, slice_y]
                elif factor>1:
                    frame = torch.zeros((d.shape[0], int(d.shape[1]*factor), int(d.shape[2]*factor)))
                    start_x = (frame.shape[1]-d.shape[1])//2
                    start_y = (frame.shape[2]-d.shape[2])//2
                    frame[:,start_x:start_x+d.shape[1], start_y:start_y+d.shape[2]] = d.clone()
                    d = frame
                data[k] = d
        return data

def get_data_augmentations(aug_config: list[dict], dtype=torch.float32) -> list:
    augs = []
    for aug_d in aug_config:
        aug_d = dict(aug_d)
        aug_name = aug_d.pop("name")
        aug = globals()[aug_name]
        if aug_name == "CastToTyped":
            # Special handling for type to enable AMP training
            islist = isinstance(aug_d["dtype"], list)
            if not islist:
                aug_d["dtype"] = [aug_d["dtype"]]
            types = [dtype if t == "dtype" else t for t in aug_d["dtype"]]
            if islist:
                aug_d["dtype"] = types
            else:
                aug_d["dtype"] = types[0]
        augs.append(aug(**aug_d))
    return augs
