import argparse
import json
import torch
import os
from tqdm import tqdm
import numpy as np

from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.networks.nets import DynUNet
from networks import MODEL_DICT

from image_dataset import get_dataset, get_post_transformation
from metrics import MetricsManager, Task

from visualizer import plot_clf_sample

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
args = parser.parse_args()

# Read config file
path = os.path.abspath(args.config_file)
with open(path) as filepath:
    config = json.load(filepath)

if not os.path.exists(config["Validation"]["save_dir"]):
    os.mkdir(config["Validation"]["save_dir"])
set_determinism(seed=0)
config["Validation"]["batch_size"]=1

task: Task = config["General"]["task"]

val_loader = get_dataset(config, 'validation')
post_pred, post_label = get_post_transformation(task, num_classes=config["Data"]["num_classes"])

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
    model = MODEL_DICT[config["General"]["model"]](num_classes=config["Data"]["num_classes"], input_channels=2 if USE_SEG_INPUT else 1).to(device)

predictions = []
tp_per_class = np.array([0 for _ in range(config["Data"]["num_classes"])])
num_pos_per_class = np.array([0 for _ in range(config["Data"]["num_classes"])])
metrics = MetricsManager(task)

checkpoint = torch.load(config["Test"]["model_path"])
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded model from epoch {checkpoint["epoch"]}')
else:
    model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    num_sample=0
    with torch.no_grad():
        step = 0
        for val_data in tqdm(val_loader, desc='Validation'):
            # if step>=config["Test"]["num_samples"]:
            #     break
            step += 1
            val_inputs, val_labels = (
                val_data["image"].to(device).float(),
                val_data["label"].to(device),
            )
            intermediate = calculate_itermediate(val_inputs)
            val_outputs: torch.Tensor = model(intermediate)
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]

            if task == Task.VESSEL_SEGMENTATION:
                pass
            else:
                pred_label=np.argmax(val_outputs[0].detach().cpu().numpy())
                true_label = np.argmax(val_labels[0].numpy())
                if pred_label==true_label:
                    tp_per_class[pred_label] += 1
                num_pos_per_class[true_label] += 1
                metrics(val_outputs, val_labels)
                
        print(f'Accuracy per class: {tp_per_class/num_pos_per_class}')
        print(f'Metrics: {metrics.aggregate_and_reset("val")}')
