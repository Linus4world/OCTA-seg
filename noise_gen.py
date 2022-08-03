
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import GaussianBlur

noise_layer_path = "/home/lkreitner/arterial-tree-generation/geometries/slab_noise.npy"
noise_layer = torch.from_numpy(np.load(noise_layer_path)).to(dtype=torch.float32).unsqueeze(0)

def scale_noise_map(noise_layer: torch.Tensor, shape):
    return torch.nn.functional.interpolate(noise_layer.unsqueeze(0), size=shape, mode='bicubic').squeeze(0)

def get_real_noise():
    img = Image.open("/home/lkreitner/arterial-tree-generation/runs/20220726_231031/art_ven_gray_z.png")
    a = np.array(img).astype(np.float32)
    img = Image.open("/home/lkreitner/arterial-tree-generation/runs/20220727_141151/art_ven_gray_z.png")
    a = np.maximum(a, np.array(img).astype(np.float32))
    img = Image.open("/home/lkreitner/arterial-tree-generation/runs/20220727_140014/art_ven_gray_z.png")
    a = np.maximum(a, np.array(img).astype(np.float32))
    a = torch.tensor(a).unsqueeze(0) / 255
    return a

def blur(img, kernel_size=(3,3)):
    kernel = torch.full((1,1,*kernel_size), 1/np.prod(kernel_size))
    blurred_img = torch.conv2d(img.unsqueeze(0), kernel, padding='same').squeeze(0)
    blurred_img = blurred_img - blurred_img.min()
    blurred_img = (blurred_img / blurred_img.max())
    return blurred_img
    # return GaussianBlur(kernel_size=kernel_size)(img)

def getGaussianNoise(shape, mean=0, std=1):
    return torch.randn(shape)*std+mean

def get_multi_level_noise(shape, l=100, kernel_size=[3,3]):
    s_0 = torch.tensor(shape)
    s = (1, *(s_0[-2:] / l).int().tolist())
    # N = torch.rand(s)*10+3
    N = torch.randn(s)*0.8+2
    N = torch.clamp(N, 0, 3)
    # N = torch.randn(s)
    # N = torch.sigmoid(N)
    # N = N / torch.max(N)
    N = torch.nn.functional.interpolate(N.unsqueeze(0), size=shape[-2:], mode='bicubic').squeeze(0)
    kernel = torch.full((1,1,*kernel_size), 1/np.prod(kernel_size))
    N = torch.conv2d(N, kernel, padding='same')
    return N

def addNoise(img, noise_layer):
    if noise_layer is not None:
        img = img.unsqueeze(0)
        if noise_layer.shape != img.shape:
            noise_layer = scale_noise_map(noise_layer, img.shape[-2:])

    N_real = get_real_noise()
    N_real = blur(N_real, kernel_size=[3,3])*0.55

    N_gauss = getGaussianNoise(img.shape, 0, 1)
    N_gauss = blur(N_gauss, (3,3)) * noise_layer
    N_gauss = N_gauss - 0.5

    multi_level_noise = get_multi_level_noise(img.shape)

    lambda_img = 0.9#np.random.uniform(0.6,1)
    lambda_N_real = 0.9
    lambda_b = 0.2
    lambda_N_gauss = 1.2
    # N_gauss[img>0]

    img = lambda_img * img

    # b = blur(img, kernel_size=(127,127))
    # b = (b-b.mean()) / (b-b.mean()).std()
    # b = b*noise_layer
    # b = b*lambda_b + 0.1

    # b[img>0]= np.maximum(0, b-img)[img>0]
    # b = np.minimum(b, np.maximum(multi_level_noise, 0))
    # b = b*np.clip(multi_level_noise,0,1)

    img = blur(img, (3,3)) * noise_layer
    img = np.maximum(img, N_real * lambda_N_real * multi_level_noise)# * b)

    img = img + N_gauss*lambda_N_gauss

    return np.clip(img, 0, 1)

    
img = Image.open("/home/lkreitner/arterial-tree-generation/dataset_22_07_22/20220722_230915/art_ven_gray_z.png")
a = np.array(img, dtype=np.float32)/255
t = torch.from_numpy(a)
t = t*3
t = torch.clamp(t,0,1)

t_noise = addNoise(t, noise_layer)
a_noise = (t_noise.numpy()*255).astype(np.uint8).squeeze()

plt.figure(figsize=(12,12))
plt.imshow(a_noise)
plt.savefig("test2.png", bbox_inches='tight')

# img = Image.open("/home/lkreitner/OCTA-seg/datasets/DRAC22/train/7.png")
# a = np.array(img)
# plt.cla()
# plt.imshow(a)
# plt.savefig("test1.png", bbox_inches='tight')

