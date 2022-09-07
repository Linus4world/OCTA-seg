import ray
from ray import tune
from ray.air import session, RunConfig, CheckpointConfig, FailureConfig
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter, SyncConfig

import argparse
import json
import os
import random
import torch

from monai.data import decollate_batch
from monai.utils import set_determinism
from networks import ResNet, MODEL_DICT, init_weights, load_intermediate_net
from monai.networks.nets import DynUNet

from image_dataset import get_dataset, get_post_transformation
from metrics import MetricsManager, Task, get_loss_function, get_loss_function_by_name
from raytune_config import OPTIMIZE_TYPE, get_raytune_config
from visualizer import Visualizer


parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--type', choices=[OPTIMIZE_TYPE.PB2, OPTIMIZE_TYPE.ASHA, OPTIMIZE_TYPE.BOHB], required=True)
parser.add_argument('--debug_mode', action="store_true")
parser.add_argument('--resume', action="store_true")
args = parser.parse_args()

# Read config file
path = os.path.abspath(args.config_file)
with open(path) as filepath:
    CONFIG = json.load(filepath)


def model_2_str(model):
    return str(model).split(' ')[1]

def training_function(config_i: dict):
    """
    The training function is used by each single trial with a different configuration.

    Paramters:
        - config_i: Dictionary that holds all necessary configuration information (deep). The current search parameters are inserted shallow. Note that the config dict must be serializable.
    """
    set_determinism(seed=config_i["General"]["seed"])
    VAL_AMP = config_i["General"]["amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)
    device = torch.device(config_i["General"]["device"])
    task: Task = config_i["General"]["task"]

    # Copy current search parameters at the correct place and overwrite defaults.
    config_i["Train"]["lr"]=config_i["lr"]
    config_i["Train"]["batch_size"]=config_i["batch_size"]
    config_i["Train"]["loss"]=config_i["loss"]
    if "model" in config_i:
        config_i["General"]["model"]=config_i["model"]

    # Load model and intermediate model
    model = MODEL_DICT[config_i["General"]["model"]]
    USE_SEG_INPUT = config_i["Train"]["model_path"] != ''
    calculate_itermediate = load_intermediate_net(
        USE_SEG_INPUT=USE_SEG_INPUT,
        model_path=config_i["Train"]["model_path"],
        num_layers=config_i["General"]["num_layers"],
        kernel_size=config_i["General"]["kernel_size"],
        num_classes=config_i["Data"]["num_classes"],
        device=device
    )
    if task == Task.AREA_SEGMENTATION:
        num_layers = config_i["General"]["num_layers"]
        kernel_size = config_i["General"]["kernel_size"]
        model: DynUNet = DynUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=config_i["Data"]["num_classes"],
            kernel_size=(3, *[kernel_size]*num_layers,3),
            strides=(1,*[2]*num_layers,1),
            upsample_kernel_size=(1,*[2]*num_layers,1),
        ).to(device)
    else:
        model: ResNet = model(num_classes=config_i["Data"]["num_classes"], input_channels=2 if USE_SEG_INPUT else 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), config_i["Train"]["lr"], weight_decay=1e-6)

    # If the trail was previously terminated or paused, it can be reloaded from a checkpoint. `Session.get_checkpoint` will automatically find the correct checkpoint.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint is not None:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            step = checkpoint['step']
            scores = checkpoint['scores']
    else:
        # If the trial starts from scratch, the weights are initialized via He initialization
        init_weights(model, init_type='kaiming')
        step=0
        scores=[]

    metrics = MetricsManager(task)
    train_loader = get_dataset(config_i, 'train', batch_size=config_i["Train"]["batch_size"])
    # Get Dataloader
    val_loader = get_dataset(config_i, 'validation')
    post_pred, post_label = get_post_transformation(task, num_classes=config_i["Data"]["num_classes"])
    loss_name = config_i["Train"]["loss"]
    loss_function = get_loss_function_by_name(loss_name, config_i)

    # TRAINING BEGINS HERE
    while True:
        model.train()
        train_loss = []
        for batch_data in train_loader:
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                intermediate = calculate_itermediate(inputs)
                outputs = model(intermediate)
                loss: torch.Tensor = loss_function(outputs, labels)
                train_loss.append(loss.item())
                labels = [post_label(i) for i in decollate_batch(labels)]
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
                metrics(y_pred=outputs, y=labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        step += 1
        metrics_dict = {
            f'train_{loss_name}': sum(train_loss)/len(train_loss)
        }
        metrics_dict.update(metrics.aggregate_and_reset(prefix="train"))
        model.eval()
        with torch.no_grad():
            val_loss = []
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device).float(),
                    val_data["label"].to(device),
                )
                intermediate = calculate_itermediate(val_inputs)
                val_outputs: torch.Tensor = model(intermediate)
                val_loss.append(loss_function(val_outputs, val_labels).item())
                
                val_labels_post = [post_label(i) for i in decollate_batch(val_labels)]
                val_outputs_post = [post_pred(i) for i in decollate_batch(val_outputs)]

                metrics(y_pred=val_outputs_post, y=val_labels_post)

            metrics_dict[f'val_{loss_name}'] = sum(val_loss)/len(val_loss)
            metrics_dict.update(metrics.aggregate_and_reset(prefix="val"))
            scores.append(metrics_dict[METRIC])
            metrics_dict[METRIC_LAST_5_MAX] = max(scores[-5:])

            # Every epoch a checkpoint is created. Tune automatically registers the checkpoint once the result is reported.
            # The Checkpoint configuration only keeps the latest and the best checkpoint on the disk and deletes all others.
            os.makedirs('my_checkpoint', exist_ok=True)
            d = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "scores": scores
            }
            Visualizer.save_tune_checkpoint('my_checkpoint/', d)
            checkpoint = Checkpoint.from_directory('my_checkpoint')
            session.report(metrics_dict, checkpoint=checkpoint)

# SEED for reproducability
CONFIG["General"]["seed"]=random.randint(0,1e6)
METRIC = 'val_QwK'
# Last 5 max is used since the validation score exhibits strong fluctuations.
METRIC_LAST_5_MAX = METRIC + "_last_5_max"
STEPS_TO_NEXT_CHECKPOINT = 5

# Load the respective scheduler and search algorithm to be used
NAME, scheduler, search_alg, num_samples = get_raytune_config(
    args.type,
    metric=METRIC_LAST_5_MAX,
    mode="max",
    seed=CONFIG["General"]["seed"]
)

# stopper = TrialPlateauStopper(METRIC_LAST_5_MAX, mode="max", num_results=40, grace_period=50)
stopper = {"training_iteration": 300}

reporter = CLIReporter(
    metric=METRIC_LAST_5_MAX,
    mode="max",
    sort_by_metric=True,
    max_report_frequency=10
)
reporter.add_metric_column(METRIC)
reporter.add_metric_column(METRIC_LAST_5_MAX)

if args.debug_mode:
    ray.init(local_mode = True)
if args.resume:
    tuner = tune.Tuner.restore("/home/lkreitner/OCTA-seg/results/raytune/ret-clf/"+NAME)
else:
    tuner = tune.Tuner(
        tune.with_resources(
            training_function,
            {"cpu": 2,"gpu": 0.5}
        ),
        # Config that is given to training function. Search parameters are added to the config by each trail individually
        param_space=CONFIG,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
            reuse_actors=True
        ),
        run_config= RunConfig(
            name=NAME,
            local_dir=CONFIG["Output"]["save_dir"],
            stop=stopper,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=METRIC,
                checkpoint_score_order="max"
            ),
            sync_config=SyncConfig(syncer=None, sync_on_checkpoint=False),
            progress_reporter=reporter,
            failure_config=FailureConfig(fail_fast=args.debug_mode)
        )
    )

results = tuner.fit()

print("Best Trail found was: ", results.get_best_result(metric=METRIC, mode='max').log_dir)