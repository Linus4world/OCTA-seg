import argparse
import json
import torch
import os
from tqdm import tqdm

from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.networks.nets import DynUNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from image_dataset import get_dataset, get_post_transformation

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
args = parser.parse_args()

# Read config file
path = os.path.abspath(args.config_file)
with open(path) as filepath:
    config = json.load(filepath)

set_determinism(seed=0)

val_loader = get_dataset(config, 'validation')
post_trans = get_post_transformation()

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()

device = torch.device(config["General"]["device"])
model = DynUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    upsample_kernel_size=(1,1,1)
).to(device)

model.load_state_dict(torch.load(config["Validation"]["model_path"]))
model.eval()

dice_metric = DiceMetric(include_background=True, reduction="mean")

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

    if config["General"]["amp"]:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

with torch.no_grad():
    for val_data in tqdm(val_loader, desc="Validating"):
        val_inputs = val_data["image"].to(device).half()
        val_labels =  val_data["label"].half()
        val_outputs = inference(val_inputs)
        val_outputs = [post_trans(i).cpu() for i in decollate_batch(val_outputs)]
        dice_metric(y_pred=val_outputs, y=val_labels)

    metric_org = dice_metric.aggregate().item()

    dice_metric.reset()


print(f"Metric on validation set: mDSC {metric_org:.4f}")