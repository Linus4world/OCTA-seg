import torch
from utils.metrics import Task

from models.networks import MODEL_DICT, init_weights


def initialize_model(config: dict, args, load_best=False):
    # Model
    num_layers = config["General"]["num_layers"]
    kernel_size = config["General"]["kernel_size"]
    device = torch.device(config["General"]["device"])
    task: Task = config["General"]["task"]
    model_path: str = config["Test"]["model_path"]
    num_classes=config["Data"]["num_classes"]
    model_name: str = config["General"]["model"]

    if task == Task.VESSEL_SEGMENTATION:
        model = MODEL_DICT[config["General"]["model"]](
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            kernel_size=(3, *[kernel_size]*num_layers,3),
            strides=(1,*[2]*num_layers,1),
            upsample_kernel_size=(1,*[2]*num_layers,1),
        ).to(device)
    elif task == Task.AREA_SEGMENTATION:
        model = MODEL_DICT[config["General"]["model"]](
            spatial_dims=2,
            in_channels=sum([True, config["Data"]["use_segmentation"], config["Data"]["use_background"]]),
            out_channels=num_classes,
            kernel_size=(3, *[kernel_size]*num_layers,3),
            strides=(1,*[2]*num_layers,1),
            upsample_kernel_size=(1,*[2]*num_layers,1),
        ).to(device)
        # Use pretrained
        # if USE_SEG_INPUT:
        #     if 'model' in checkpoint:
        #         model.load_state_dict(checkpoint['model'], strict=False)
        #     else:
        #         # filter unnecessary keys
        #         pretrained_dict = {k: v for k, v in checkpoint.items() if
        #                             (k in model.state_dict().keys()) and (model.state_dict()[k].shape == checkpoint[k].shape)}
        #         model.load_state_dict(pretrained_dict, strict=False)
    else:
        model = MODEL_DICT[model_name](num_classes=num_classes, input_channels=sum([True, config["Data"]["use_segmentation"], config["Data"]["use_background"]])).to(device)

    if hasattr(args, "start_epoch") and args.start_epoch>0:
        checkpoint = torch.load(model_path.replace('best_model', 'latest_model'))
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"], weight_decay=1e-6)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint from epoch", checkpoint['epoch'])
    elif load_best:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer=None
        print("Loaded checkpoint from epoch", checkpoint['epoch'])
    else:
        activation = 'relu' if model_name.startswith("resnet") else 'leaky_relu'
        init_weights(model, init_type='kaiming', nonlinearity=activation)
        optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"], weight_decay=1e-6)
    return model, optimizer