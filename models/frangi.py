from skimage.filters import frangi
import numpy as np
import torch

class Frangi():
    def __init__(self, **kwargs) -> None:
        pass
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Computes the frangi filter image. Binarization needs to be applied afterwards.
        Implementation uses https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.frangi

        Args:
            img (torch.Tensor): Image tensor of shape [B,C,H,W] scaled to [0,1]

        Returns:
            torch.Tensor: filtered image tensor of shape [B,C,H,W] scaled to [0,1]
        """
        assert img.shape[0]==1
        frangi_img = img.squeeze().numpy() * 255
        frangi_img = frangi(frangi_img, sigmas = (0.5,2,0.5), alpha=1, beta=15, black_ridges=False)
        frangi_img = torch.tensor(frangi_img).view(img.shape)
        # frangi_img = (frangi_img>=threshold).astype(np.float32)
        return frangi_img
    def eval(self):
        pass
    def train(self):
        pass

if __name__ == "__main__":
    from monai.transforms import RemoveSmallObjects
    from PIL import Image

    # path = "/home/lkreitner/OCTA-seg/datasets/Edinburgh/original_images/2.png"
    # path = "/home/shared/Data/ROSE/ROSE-1/SVC_DVC/train/img/06.png"
    path = "/home/shared/Data/OCTA-500/Labels and projection maps/OCTA_3M/Projection Maps/OCTA(ILM_OPL)/10301.bmp"

    name = path.split("/")[-1].split(".")[0]
    img = np.array(Image.open(path).convert('L')).astype(np.float32)

    # removeSmallObjects = RemoveSmallObjects(8) # Edinburgh
    # removeSmallObjects = RemoveSmallObjects(32) # ROSE
    removeSmallObjects = RemoveSmallObjects(64) # OCTA-500

    frangi_img = frangi(img, sigmas = (0.5,2,0.5), alpha=1, beta=15, black_ridges=False)

    # threshold = threshold_local(img, 15, method="gaussian")/255 # adpative
    # threshold = 0.1 # Edinburgh
    # threshold = 0.3 # ROSE
    # threshold = 0.55 # OCTA-500
    threshold = 0.1

    frangi_img = (frangi_img>=threshold).astype(np.float32)
    frangi_img = removeSmallObjects(frangi_img)
    frangi_img = np.clip(frangi_img*255, 0, 255).astype(np.uint8)
    Image.fromarray(frangi_img).save(f"{name}-frangi-{threshold}.png")