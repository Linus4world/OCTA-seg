import argparse
import json
import os
import torch
import datetime
import shutil
import csv

# from torch.utils.data import Dataset
from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from monai.networks.nets import DynUNet
import time
from tqdm import tqdm

from image_dataset import get_dataset, get_post_transformation
from visualizer import plot_losses_and_metrics, plot_sample

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
args = parser.parse_args()

# Read config file
path = os.path.abspath(args.config_file)
with open(path) as filepath:
    config = json.load(filepath)


set_determinism(seed=0)

train_loader = get_dataset(config, 'train')
val_loader = get_dataset(config, 'validation')

max_epochs = config["Train"]["epochs"]
val_interval = config["Train"]["val_interval"]
VAL_AMP = config["General"]["amp"]

device = torch.device(config["General"]["device"])

# Model
model = DynUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    upsample_kernel_size=(1,1,1)
).to(device)

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = get_post_transformation()

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()

# define inference method
def inference(input):

    def _compute(input):
        return sliding_window_inference(
            roi_size=(256,256),
            inputs=input,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

save_dir = os.path.join(config["Output"]["save_dir"], datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
os.mkdir(save_dir)
shutil.copyfile(args.config_file, os.path.join(save_dir, 'config.json'))

log_file_path = os.path.join(save_dir, 'metrics.csv')
with open(log_file_path, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Average Loss", "Val Mean Dice Score"])


# TRAINING BEGINS HERE

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []

total_start = time.time()
epoch_tqdm = tqdm(range(max_epochs), desc="epoch")
for epoch in epoch_tqdm:
    model.train()
    epoch_loss = 0
    step = 0
    mini_batch_tqdm = tqdm(train_loader, leave=False)
    for batch_data in mini_batch_tqdm:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device).half(),
            batch_data["label"].to(device).half(),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        mini_batch_tqdm.set_description(f'train_loss: {loss.item():.4f}')
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    epoch_tqdm.set_description(f'average loss: {epoch_loss:.4f}')

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in tqdm(val_loader, desc='Validation', leave=False):
                val_inputs, val_labels = (
                    val_data["image"].to(device).half(),
                    val_data["label"].to(device).half(),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "best_metric_model.pth"),
                )
            with open(log_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, epoch_loss, metric])
            plot_losses_and_metrics(
                epoch_loss_values=epoch_loss_values,
                metric_values=metric_values,
                val_interval=val_interval,
                save_dir=save_dir
            )
            plot_sample(val_inputs[0], val_outputs[0], val_labels[0], save_dir=save_dir)
total_time = time.time() - total_start

print(f'Finished training after {str(datetime.timedelta(seconds=total_time))}')