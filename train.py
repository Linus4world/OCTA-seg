import argparse
import json
import os
import torch
import datetime

# from torch.utils.data import Dataset
from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.networks.nets import DynUNet
from networks import resnet18, resnet50
import time
from tqdm import tqdm

from image_dataset import get_dataset, get_post_transformation
from metrics import MetricsManager, Task, get_loss_function
from visualizer import Visualizer, extract_vessel_graph_features

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--start_epoch', type=int, default=0)
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
set_determinism(seed=0)
model_path: str = config["Test"]["model_path"]
visualizer = Visualizer(config, args.start_epoch>0)

train_loader = get_dataset(config, 'train')
val_loader = get_dataset(config, 'validation')
post_pred, post_label = get_post_transformation(task, num_classes=config["Data"]["num_classes"])

# Model
num_layers = config["General"]["num_layers"]
kernel_size = config["General"]["kernel_size"]
pre_model = None
if task == Task.VESSEL_SEGMENTATION or task == Task.AREA_SEGMENTATION:
    model = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=config["Data"]["num_classes"],
        kernel_size=(3, *[kernel_size]*num_layers,3),
        strides=(1,*[2]*num_layers,1),
        upsample_kernel_size=(1,*[2]*num_layers,1),
    ).to(device)
elif task == Task.VESSEL_SEGMENTATION_THEN_RETINOPATHY_CLASSIFICATION:
    pre_model =  model = DynUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=config["Data"]["num_classes"],
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
    model = resnet18(num_classes=config["Data"]["num_classes"], input_channels=2).to(device)
else:
    # model = DenseNet169(spatial_dims=2, in_channels=1, out_channels=config["Data"]["num_classes"], dropout_prob=config["Train"]["dropout_prob"]).to(device)
    model = resnet18(num_classes=config["Data"]["num_classes"]).to(device)
if args.start_epoch>0:
    checkpoint = torch.load(model_path.replace('best_model', 'latest_model'))
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"])#, weight_decay=1e-5)

with torch.no_grad():
    visualizer.save_model_architecture(model, next(iter(train_loader))["image"].to(device=device, dtype=torch.float32))

loss_function = get_loss_function(task, config)
def schedule(step: int):
    if step < max_epochs - config["Train"]["epochs_decay"]:
        return 1
    else:
        return (max_epochs-step) * (1/max(1,config["Train"]["epochs_decay"]))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
metrics = MetricsManager(task)


# TRAINING BEGINS HERE
best_metric = -1
best_metric_epoch = -1

total_start = time.time()
epoch_tqdm = tqdm(range(args.start_epoch,max_epochs), desc="epoch")
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
            if pre_model is not None:
                intermediate = pre_model(inputs)
                intermediate = torch.stack([torch.tensor(extract_vessel_graph_features(inter)) for inter in intermediate]).to(device=device, dtype=torch.float16 if VAL_AMP else torch.float32)
                intermediate = torch.cat([inputs,intermediate.unsqueeze(1)], dim=1)
            else:
                intermediate = inputs
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
    if task == Task.VESSEL_SEGMENTATION or task == Task.AREA_SEGMENTATION:
        visualizer.plot_sample(inputs[0], outputs[0], labels[0], suffix='train')
    else:
        visualizer.plot_clf_sample(inputs, outputs, labels, suffix='train')

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
                    if pre_model is not None:
                        intermediate = pre_model(val_inputs)
                        intermediate = torch.stack([torch.tensor(extract_vessel_graph_features(inter)) for inter in intermediate]).to(device=device, dtype=torch.float16 if VAL_AMP else torch.float32)
                        intermediate = torch.cat([val_inputs,intermediate.unsqueeze(1)], dim=1)
                    else:
                        intermediate = val_inputs
                    val_outputs = model(intermediate)
                    val_loss += loss_function(val_outputs, val_labels).item()
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    metrics(y_pred=val_outputs, y=val_labels)

            epoch_metrics["loss"]["val_loss"] = val_loss/step
            epoch_metrics["metric"].update(metrics.aggregate_and_reset(prefix="val"))

            metric_comp =  epoch_metrics["metric"][metrics.get_comp_metric('val')]

            visualizer.save_model(model, optimizer, epoch, 'latest')
            if metric_comp > best_metric:
                best_metric = metric_comp
                best_metric_epoch = epoch + 1
                visualizer.save_model(model, optimizer, epoch, 'best')

            visualizer.plot_losses_and_metrics(epoch_metrics, epoch)
            if epoch%config["Output"]["save_interval"] == 0:
                if task == Task.VESSEL_SEGMENTATION or task == Task.AREA_SEGMENTATION:
                    visualizer.plot_sample(val_inputs[0], val_outputs[0], val_labels[0], suffix= None if best_metric>metric_comp else 'best')
                else:
                    visualizer.plot_clf_sample(val_inputs, val_outputs, val_labels, suffix= None if best_metric>metric_comp else 'best')
    visualizer.log_model_params(model, epoch)

total_time = time.time() - total_start

print(f'Finished training after {str(datetime.timedelta(seconds=total_time))}. Best metric: {best_metric} at epoch: {best_metric_epoch}')