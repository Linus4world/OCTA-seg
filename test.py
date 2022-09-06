import argparse
import json
import torch
import os
from tqdm import tqdm
import numpy as np

from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.networks.nets import DynUNet
from networks import resnet18

from image_dataset import get_dataset, get_post_transformation
from metrics import Task

from visualizer import plot_sample, save_prediction_csv

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
args = parser.parse_args()

# Read config file
path = os.path.abspath(args.config_file)
with open(path) as filepath:
    config = json.load(filepath)

if not os.path.exists(config["Test"]["save_dir"]):
    os.mkdir(config["Test"]["save_dir"])
set_determinism(seed=0)

task: Task = config["General"]["task"]

test_loader = get_dataset(config, 'test')
post_trans, _ = get_post_transformation(task)

VAL_AMP = config["General"]["amp"]
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)
device = torch.device(config["General"]["device"])

USE_SEG_INPUT = config["Train"]["model_path"] != ''

# Model
num_layers = config["General"]["num_layers"]
kernel_size = config["General"]["kernel_size"]
if USE_SEG_INPUT:
    pre_model = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        kernel_size=(3, *[kernel_size]*num_layers,3),
        strides=(1,*[2]*num_layers,1),
        upsample_kernel_size=(1,*[2]*num_layers,1),
    ).to(device)
    checkpoint = torch.load(config["Train"]["model_path"])
    if hasattr(checkpoint, 'model'):
        pre_model.load_state_dict(checkpoint['model'])
    else:
        pre_model.load_state_dict(checkpoint)
    pre_model.eval()
    post_itermediate, _ = get_post_transformation(Task.VESSEL_SEGMENTATION, num_classes=config["Data"]["num_classes"])

    def calculate_itermediate(inputs: torch.Tensor):
        with torch.no_grad():
            intermediate = pre_model(inputs)
            intermediate = torch.stack([post_itermediate(inter) for inter in intermediate])
            intermediate = torch.cat([inputs,intermediate], dim=1)
            return intermediate
else:
    def calculate_itermediate(inputs: torch.Tensor):
        return inputs

if task == Task.VESSEL_SEGMENTATION:
    model = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=config["Data"]["num_classes"],
        kernel_size=(3, *[kernel_size]*num_layers,3),
        strides=(1,*[2]*num_layers,1),
        upsample_kernel_size=(1,*[2]*num_layers,1),
    ).to(device)
elif task == Task.AREA_SEGMENTATION:
    model = DynUNet(
        spatial_dims=2,
        in_channels= 2 if USE_SEG_INPUT else 1,
        out_channels=config["Data"]["num_classes"],
        kernel_size=(3, *[kernel_size]*num_layers,3),
        strides=(1,*[2]*num_layers,1),
        upsample_kernel_size=(1,*[2]*num_layers,1),
    ).to(device)
else:
    model = resnet18(num_classes=config["Data"]["num_classes"], input_channels=2 if USE_SEG_INPUT else 1).to(device)

predictions = []
checkpoint = torch.load(config["Test"]["model_path"])
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded model from epoch {checkpoint["epoch"]}')
else:
    model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    num_sample=0
    for test_data in tqdm(test_loader, desc="Testing", total=min(len(test_loader), config["Test"]["num_samples"])):
        if num_sample>=config["Test"]["num_samples"]:
            break
        num_sample+=1
        if task == Task.VESSEL_SEGMENTATION:
            val_inputs = test_data.to(device)
        else:
            val_inputs = test_data["image"].to(device)
        with torch.cuda.amp.autocast():
            intermediate = calculate_itermediate(val_inputs)
            val_outputs = model(intermediate)
        val_outputs = [post_trans(i).cpu() for i in decollate_batch(val_outputs)]

        if task == Task.VESSEL_SEGMENTATION:
            # clean_seg = extract_vessel_graph_features(val_outputs[0], config["Test"]["save_dir"], config["Voreen"], number=num_sample)
            # graph_file = os.path.join(config["Test"]["save_dir"], f'sample_{num_sample}_graph.json')
            # graph_img = graph_file_to_img(graph_file, val_outputs[0].shape[-2:])
            plot_sample(config["Test"]["save_dir"], val_inputs[0], val_outputs[0], None, number=num_sample)
        else:
            for i, path in enumerate(test_data["path"]):
                predictions.append([str(path.split('/')[-1]), np.argmax(val_outputs[i].numpy()), *val_outputs[i].numpy().tolist()])
                difference = sum(predictions[-1][2:])-1
                if difference!=0:
                    idx = np.argsort(predictions[-1][2:])[-2]+2
                    predictions[-1][idx] = predictions[-1][idx]-difference
    if task == Task.RETINOPATHY_CLASSIFICATION or Task.IMAGE_QUALITY_CLASSIFICATION:
        save_prediction_csv(config["Test"]["save_dir"], predictions)
        # plot_sample(config["Test"]["save_dir"], val_inputs[0], torch.tensor(clean_seg), graph_img, number=num_sample)
