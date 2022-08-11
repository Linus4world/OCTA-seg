import argparse
import json
import os
import torch
import datetime

# from torch.utils.data import Dataset
from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.metrics import DiceMetric, MeanIoU

from monai.networks.nets import DynUNet
import time
from tqdm import tqdm
from cl_dice_loss import clDiceLoss

from image_dataset import get_dataset, get_post_transformation
from metrics import MetricsManager, Task
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
dtype = torch.float16 if VAL_AMP else torch.float32
device = torch.device(config["General"]["device"])
task: Task = config["General"]["task"]
set_determinism(seed=0)
visualizer = Visualizer(config, args.config_file)

train_loader = get_dataset(config, 'train')
val_loader = get_dataset(config, 'validation')
post_trans = get_post_transformation()

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
visualizer.save_model_architecture(model, next(iter(train_loader))["image"].to(device=device))

loss_function = clDiceLoss(alpha=config["Train"]["lambda_cl_dice"], sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
metrics = MetricsManager(task)
# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)



# TRAINING BEGINS HERE
best_metric = -1
best_metric_epoch = -1

total_start = time.time()
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
            batch_data["image"].to(device, dtype=dtype),
            batch_data["label"].to(device, dtype=dtype),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            outputs = [post_trans(i) for i in decollate_batch(outputs)]
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
                    val_data["image"].to(device).half(),
                    val_data["label"].to(device).half(),
                )
                with torch.cuda.amp.autocast():
                    val_outputs = model(val_inputs)
                    val_loss += loss_function(val_outputs, val_labels).item()
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                metrics(y_pred=val_outputs, y=val_labels)

            epoch_metrics["loss"]["val_loss"] = val_loss/step
            epoch_metrics["metric"].update(metrics.aggregate_and_reset(prefix="val"))

            metric_comp =  epoch_metrics["metric"][metrics.get_comp_metric('val')]

            if metric_comp > best_metric:
                best_metric = metric_comp
                best_metric_epoch = epoch + 1
                visualizer.save_model(model)

            visualizer.plot_losses_and_metrics(epoch_metrics, epoch)
            visualizer.plot_sample(val_inputs[0], val_outputs[0], val_labels[0], number= None if best_metric>metric_comp else 'best')
    visualizer.log_model_params(model, epoch)

total_time = time.time() - total_start

print(f'Finished training after {str(datetime.timedelta(seconds=total_time))}. Best metric: {best_metric} at epoch: {best_metric_epoch}')