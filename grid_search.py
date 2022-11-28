import argparse
import json
import os
import torch
from itertools import product
import copy

# from torch.utils.data import Dataset
from monai.data import decollate_batch
from monai.utils import set_determinism
import yaml
from models.networks import ResNet, resnet18, resnet50
from monai.networks.nets import DynUNet
from tqdm import tqdm

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager, Task, get_loss_function
from utils.visualizer import Visualizer

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

max_epochs = config["Train"]["epochs"]
val_interval = config["Train"]["val_interval"]
VAL_AMP = config["General"]["amp"]
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)
device = torch.device(config["General"]["device"])
task: Task = config["General"]["task"]

# Get Dataloader
val_loader = get_dataset(config, 'validation')
post_pred, post_label = get_post_transformation(task, num_classes=config["Data"]["num_classes"])

loss_name, loss_function = get_loss_function(task, config)

set_determinism(seed=0)

def model_2_str(model):
    return str(model).split(' ')[1]

search_space = {
    "model": [DynUNet],
    "lr": [1e-4, 1e-3, 3e-3],
    "batch_size": [2,4,8]
}

def train_model_i(config_i: dict, config: dict):
    config["GRID_SEARCH"] = dict(config_i)
    config["GRID_SEARCH"]["model"] = model_2_str(params[0])
    visualizer = Visualizer(config)
    if task == Task.AREA_SEGMENTATION:
        num_layers = config["General"]["num_layers"]
        kernel_size = config["General"]["kernel_size"]
        model: DynUNet = DynUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=config["Data"]["num_classes"],
            kernel_size=(3, *[kernel_size]*num_layers,3),
            strides=(1,*[2]*num_layers,1),
            upsample_kernel_size=(1,*[2]*num_layers,1),
        ).to(device)
    else:
        model: ResNet = config_i["model"](num_classes=config["Data"]["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config_i["lr"])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    metrics = MetricsManager(task)
    train_loader = get_dataset(config, 'train', batch_size=config_i["batch_size"])

    # TRAINING BEGINS HERE
    best_metric = -1

    epoch_tqdm = tqdm(range(max_epochs), desc="epoch")
    for epoch in epoch_tqdm:
        epoch_metrics = dict()
        model.train()
        epoch_loss = 0
        step = 0
        mini_batch_tqdm = tqdm(train_loader, leave=False)
        for batch_data in mini_batch_tqdm:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                labels = [post_label(i) for i in decollate_batch(labels)]
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
                metrics(y_pred=outputs, y=labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            mini_batch_tqdm.set_description(f'train_{loss_name}: {loss.item():.4f}')
        # lr_scheduler.step()

        epoch_loss /= step
        epoch_metrics["loss"] = {
            f"train_{loss_name}": epoch_loss
        }
        epoch_metrics["metric"] = metrics.aggregate_and_reset(prefix="train")
        epoch_tqdm.set_description(f'avg train loss: {epoch_loss:.4f}')

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                step = 0
                for val_data in tqdm(val_loader, desc='Validation', leave=False):
                    step += 1
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    with torch.cuda.amp.autocast():
                        val_outputs = model(val_inputs)
                        val_loss += loss_function(val_outputs, val_labels).item()
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        metrics(y_pred=val_outputs, y=val_labels)

                epoch_metrics["loss"][f"val_f{loss_name}"] = val_loss/step
                epoch_metrics["metric"].update(metrics.aggregate_and_reset(prefix="val"))

                metric_comp =  epoch_metrics["metric"][metrics.get_comp_metric('val')]
                visualizer.save_model(model, optimizer, epoch, 'latest')
                if metric_comp > best_metric:
                    best_metric = metric_comp
                    visualizer.save_model(model, optimizer, epoch, 'best')
                visualizer.plot_losses_and_metrics(epoch_metrics, epoch)
    config_i["model"] = model_2_str(params[0])
    visualizer.save_hyperparams(config_i, {metrics.get_comp_metric('val'): best_metric})

param_values = [v for v in search_space.values()]
t = list(product(*param_values))
main_loop = tqdm(t)
for params in main_loop:
    params_titles = [model_2_str(params[0]), *params[1:]]
    main_loop.set_description(','.join([str(p) for p in params_titles]))
    train_model_i({k: p for k,p in zip(search_space.keys(), params)}, copy.deepcopy(config))