import argparse
import json
import torch
import os
from tqdm import tqdm
import numpy as np

from monai.data import decollate_batch
import yaml
from models.model import initialize_model, initialize_optimizer

from image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager, Task

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
args = parser.parse_args()

# Read config file
path: str = os.path.abspath(args.config_file)
with open(path, "r") as stream:
    if path.endswith(".json"):
        config = json.load(stream)
    else:
        config = yaml.safe_load(stream)

config["Validation"]["batch_size"]=1

task: Task = config["General"]["task"]

val_loader = get_dataset(config, 'validation')
post_pred, post_label = get_post_transformation(task, num_classes=config["Data"]["num_classes"])

VAL_AMP = config["General"]["amp"]
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)
device = torch.device(config["General"]["device"])

model = initialize_model(config)
optimizer = initialize_optimizer(model, config, args, load_best=True)

metrics = MetricsManager(task)
predictions = []
num_classes = config["Data"]["num_classes"] if config["Data"]["num_classes"]>1 else 3
tp_per_class = np.array([0 for _ in range(num_classes)])
num_pos_per_class = np.array([0 for _ in range(num_classes)])

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
            val_outputs: torch.Tensor = model(val_inputs)
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]

            if task == Task.VESSEL_SEGMENTATION:
                pass
            elif task == Task.IMAGE_QUALITY_CLASSIFICATION or task == Task.RETINOPATHY_CLASSIFICATION:
                pred_label=np.argmax(val_outputs[0].detach().cpu().numpy())
                true_label = np.argmax(val_labels[0].numpy())
                if pred_label==true_label:
                    tp_per_class[pred_label] += 1
                num_pos_per_class[true_label] += 1
                print(f'Accuracy per class: {tp_per_class/num_pos_per_class}')
            metrics(val_outputs, val_labels)
                
        metrics = {k: str(round(v, 4)) for k,v in metrics.aggregate_and_reset("val").items()}
        print(f'Metrics: {metrics}')
