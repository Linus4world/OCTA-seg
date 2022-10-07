import argparse
import json
import torch
import os
from tqdm import tqdm
import numpy as np

from monai.data import decollate_batch
from monai.utils import set_determinism
from models.model import initialize_model

from image_dataset import get_dataset, get_post_transformation
from utils.masks_to_nii import masks2nii
from utils.metrics import Task

from utils.visualizer import plot_sample, plot_single_image, save_prediction_csv

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
post_trans, _ = get_post_transformation(task, num_classes=config["Data"]["num_classes"])

VAL_AMP = config["General"]["amp"]
# use amp to accelerate training
device = torch.device(config["General"]["device"])

model, optimizer = initialize_model(config, args, load_best=True)
predictions = []

model.eval()
with torch.no_grad():
    num_sample=0
    for test_data in tqdm(test_loader, desc="Testing", total=min(len(test_loader), config["Test"]["num_samples"])):
        if num_sample>=config["Test"]["num_samples"]:
            break
        num_sample+=1
        val_inputs = test_data["image"].to(device).float()
        val_outputs = model(val_inputs)
        val_outputs = [post_trans(i).cpu() for i in decollate_batch(val_outputs)]

        if task == Task.VESSEL_SEGMENTATION:
            # clean_seg = extract_vessel_graph_features(val_outputs[0], config["Test"]["save_dir"], config["Voreen"], number=num_sample)
            # graph_file = os.path.join(config["Test"]["save_dir"], f'sample_{num_sample}_graph.json')
            # graph_img = graph_file_to_img(graph_file, val_outputs[0].shape[-2:])
            plot_sample(config["Test"]["save_dir"], val_inputs[0], val_outputs[0], None, test_data["path"][0], suffix=f"{num_sample}", full_size=True)
            # plot_single_image(config["Test"]["save_dir"], val_inputs[0], num_sample*10+1)
            # plot_single_image(config["Test"]["save_dir"], val_outputs[0], test_data["path"][0].split("/")[-1])
            # diff = val_inputs[0].clone()
            # diff[val_outputs[0]==1]=0
            # plot_single_image(config["Test"]["save_dir"], diff, num_sample*10+3)
        elif task == Task.AREA_SEGMENTATION:
            for i in range(len(val_outputs[0])):
                dir = os.path.join(config["Test"]["save_dir"], str(i))
                if not os.path.exists(dir):
                    os.mkdir(dir)
                plot_single_image(dir, val_outputs[0][i], test_data["path"][0].split("/")[-1])
        else:
            for i, path in enumerate(test_data["path"]):
                predictions.append([str(path.split('/')[-1]), np.argmax(val_outputs[i].numpy()), *val_outputs[i].numpy().tolist()])
                difference = sum(predictions[-1][2:])-1
                if difference!=0:
                    idx = np.argsort(predictions[-1][2:])[-2]+2
                    predictions[-1][idx] = predictions[-1][idx]-difference
    if task == Task.RETINOPATHY_CLASSIFICATION or task == Task.IMAGE_QUALITY_CLASSIFICATION:
        save_prediction_csv(config["Test"]["save_dir"], predictions)
    elif task == Task.AREA_SEGMENTATION:
        for i in range(len(val_outputs[0])):
            dir = os.path.join(config["Test"]["save_dir"], str(i))
            masks2nii(dir)
