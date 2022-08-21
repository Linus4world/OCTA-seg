import argparse
import json
import torch
import os
from tqdm import tqdm

from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.networks.nets import DynUNet

from image_dataset import get_dataset, get_post_transformation
from metrics import Task

from visualizer import extract_vessel_graph_features, graph_file_to_img, plot_sample

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

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()

device = torch.device(config["General"]["device"])
# Model
num_layers = config["General"]["num_layers"]
kernel_size = config["General"]["kernel_size"]
model = DynUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    kernel_size=(3, *[kernel_size]*num_layers,3),
    strides=(1,*[2]*num_layers,1),
    upsample_kernel_size=(1,*[2]*num_layers,1),
).to(device)

model.load_state_dict(torch.load(config["Test"]["model_path"]))
model.eval()

with torch.no_grad():
    num_sample=0
    for test_data in tqdm(test_loader, desc="Testing", total=min(len(test_loader), config["Test"]["num_samples"])):
        if num_sample>=config["Test"]["num_samples"]:
            break
        num_sample+=1
        val_inputs = test_data.to(device)
        if config["General"]["amp"]:
            with torch.cuda.amp.autocast():
                val_outputs = model(val_inputs)
        else:
            val_outputs = model(val_inputs)
        val_outputs = [post_trans(i).cpu() for i in decollate_batch(val_outputs)]
        extract_vessel_graph_features(val_outputs[0], config["Test"]["save_dir"], config["Voreen"], number=num_sample)
        graph_file = os.path.join(config["Test"]["save_dir"], f'sample_{num_sample}_graph.json')
        graph_img = graph_file_to_img(graph_file)
        
        plot_sample(config["Test"]["save_dir"], val_inputs[0], val_outputs[0], graph_img, number=num_sample)
