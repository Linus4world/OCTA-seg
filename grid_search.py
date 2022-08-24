import argparse
import json
import os
import torch
from itertools import product
import copy

# from torch.utils.data import Dataset
from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet
from tqdm import tqdm

from image_dataset import get_dataset, get_post_transformation
from metrics import MetricsManager, Task, get_loss_function
from visualizer import Visualizer

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
args = parser.parse_args()

# Read config file
path = os.path.abspath(args.config_file)
with open(path) as filepath:
    config = json.load(filepath)

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

loss_function = get_loss_function(task, config)

set_determinism(seed=0)

def model_2_str(model):
    return str(model).split(' ')[1]

search_space = {
    "model": [DenseNet121, DenseNet169],
    "dropout_prob": [0, 0.1],
    "weight_decay": [0, 1e-5],
    "lr": [2e-4, 1e-4, 5e-5],
    "batch_size": [4,2]
}

def train_model_i(config_i: dict, config: dict):
    config["GRID_SEARCH"] = dict(config_i)
    config["GRID_SEARCH"]["model"] = str(params[0]).split('.')[-1].split('\'')[0]
    visualizer = Visualizer(config)
    model: DenseNet = config_i["model"](spatial_dims=2, in_channels=1, out_channels=config["Data"]["num_classes"], dropout_prob=config_i["dropout_prob"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config_i["lr"], weight_decay=config_i["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
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
            mini_batch_tqdm.set_description(f'train_loss: {loss.item():.4f}')
        lr_scheduler.step()

        epoch_loss /= step
        epoch_metrics["loss"] = {
            "train_loss": epoch_loss
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

                epoch_metrics["loss"]["val_loss"] = val_loss/step
                epoch_metrics["metric"].update(metrics.aggregate_and_reset(prefix="val"))

                metric_comp =  epoch_metrics["metric"][metrics.get_comp_metric('val')]
                visualizer.save_model(model, 'latest')
                if metric_comp > best_metric:
                    best_metric = metric_comp
                    visualizer.save_model(model, 'best')
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