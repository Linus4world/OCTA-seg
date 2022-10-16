import json
import torch
import os
from tqdm import tqdm
import numpy as np

from monai.data import decollate_batch
from monai.utils import set_determinism
import yaml
from models.model import initialize_model, initialize_optimizer

from image_dataset import get_dataset, get_post_transformation
from utils.masks_to_nii import masks2nii
from utils.metrics import Task

from utils.visualizer import save_prediction_csv, plot_single_image

config_files = [
    "/home/lkreitner/OCTA-seg/results/area-seg/cross_val/pre_ves_seg/20220922_122224/config.json",
    "/home/lkreitner/OCTA-seg/results/area-seg/cross_val/pre_ves_seg/20220922_122356/config.json",
    "/home/lkreitner/OCTA-seg/results/area-seg/cross_val/pre_ves_seg/20220922_122427/config.json",
    "/home/lkreitner/OCTA-seg/results/area-seg/cross_val/pre_ves_seg/20220922_232631/config.json",
    "/home/lkreitner/OCTA-seg/results/area-seg/cross_val/pre_ves_seg/20220923_173815/config.json"
]

predictions = dict()
for config_file in tqdm(config_files):

    # Read config file
    path: str = os.path.abspath(config_file)
    with open(path, "r") as stream:
        if path.endswith(".json"):
            config = json.load(stream)
        else:
            config = yaml.safe_load(stream)

    if not os.path.exists(config["Test"]["save_dir"]):
        os.mkdir(config["Test"]["save_dir"])
    set_determinism(seed=0)

    task: Task = config["General"]["task"]

    test_loader = get_dataset(config, 'test')
    post_trans, _ = get_post_transformation(task, num_classes=config["Data"]["num_classes"])

    VAL_AMP = config["General"]["amp"]
    # use amp to accelerate training
    device = torch.device(config["General"]["device"])

    model = initialize_model(config)
    optimizer = initialize_optimizer(model, config, None, load_best=True)
    model.eval()
    with torch.no_grad():
        num_sample=0
        for test_data in tqdm(test_loader, desc="Testing", total=min(len(test_loader), config["Test"]["num_samples"])):
            if num_sample>=config["Test"]["num_samples"]:
                break
            num_sample+=1
            val_inputs = test_data["image"].to(device).float()
            val_outputs = model(val_inputs)
            if task == Task.AREA_SEGMENTATION:
                val_outputs = [post_trans.transforms[0](i).cpu() for i in decollate_batch(val_outputs)]
            else:
                val_outputs = [post_trans(i).cpu() for i in decollate_batch(val_outputs)]
            for i, path in enumerate(test_data["path"]):
                key = str(path.split('/')[-1])
                predictions[key] = predictions[key] + val_outputs[i].numpy() if key in predictions else val_outputs[i].numpy()

if task == Task.RETINOPATHY_CLASSIFICATION or task == Task.IMAGE_QUALITY_CLASSIFICATION:
    predictions = {k: v/len(config_files) for k,v in predictions.items()}
    final_pred = []
    for k,v in predictions.items():
        final_pred.append([k, np.argmax(v), *v.tolist()])
        difference = sum(final_pred[-1][2:])-1
        if difference!=0:
            idx = np.argsort(final_pred[-1][2:])[-2]+2
            final_pred[-1][idx] = final_pred[-1][idx]-difference
    save_prediction_csv(config["Test"]["save_dir"], final_pred)

elif task == Task.AREA_SEGMENTATION:
    for k, v in predictions.items():
        seg = post_trans.transforms[1](v/len(config_files))
        for i in range(len(seg)):
            dir = os.path.join(config["Test"]["save_dir"], str(i))
            if not os.path.exists(dir):
                os.mkdir(dir)
            plot_single_image(dir, seg[i], k)

    for i in range(len(val_outputs[0])):
        dir = os.path.join(config["Test"]["save_dir"], str(i))
        masks2nii(dir)
