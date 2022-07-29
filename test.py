import json
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.networks.nets import DynUNet

from image_dataset import get_dataset, get_post_transformation

from visualizer import plot_sample

# Read config file
path = os.path.abspath("/home/lkreitner/OCTA-seg/configs/config.json")
with open(path) as filepath:
    config = json.load(filepath)

if not os.path.exists(config["Test"]["save_dir"]):
    os.mkdir(config["Test"]["save_dir"])
set_determinism(seed=0)

test_loader = get_dataset(config, 'test')
post_trans = get_post_transformation()

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()

device = torch.device(config["General"]["device"])
model = DynUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    upsample_kernel_size=(1,1,1)
).to(device)

model.load_state_dict(torch.load(config["Test"]["model_path"]))
model.eval()

with torch.no_grad():
    num_sample=0
    for test_data in tqdm(test_loader, desc="Testing", total=min(len(test_loader), config["Test"]["num_samples"])):
        if num_sample>=config["Test"]["num_samples"]:
            break
        num_sample+=1
        val_inputs = test_data.to(device).half()
        if config["General"]["amp"]:
            with torch.cuda.amp.autocast():
                val_outputs = model(val_inputs)
        else:
            val_outputs = model(val_inputs)
        val_outputs = [post_trans(i).cpu() for i in decollate_batch(val_outputs)]
        plot_sample(config["Test"]["save_dir"], val_inputs[0], val_outputs[0], save=True, number=num_sample)
