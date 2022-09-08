import torch
from utils.metrics import Task

from models.networks import MODEL_DICT, init_weights, load_intermediate_net


def initialize_model(config: dict, args):
    # Model
    num_layers = config["General"]["num_layers"]
    kernel_size = config["General"]["kernel_size"]
    device = torch.device(config["General"]["device"])
    task: Task = config["General"]["task"]
    model_path: str = config["Test"]["model_path"]
    USE_SEG_INPUT = config["Train"]["model_path"] != ''
    num_classes=config["Data"]["num_classes"]

    calculate_itermediate = load_intermediate_net(
        USE_SEG_INPUT=USE_SEG_INPUT,
        model_path=model_path,
        num_layers=num_layers,
        kernel_size=kernel_size,
        num_classes=num_classes,
        device=device
    )

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
            in_channels=2 if USE_SEG_INPUT else 1,
            out_channels=num_classes,
            kernel_size=(3, *[kernel_size]*num_layers,3),
            strides=(1,*[2]*num_layers,1),
            upsample_kernel_size=(1,*[2]*num_layers,1),
        ).to(device)
        init_weights(model, init_type='kaiming')
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
        model = MODEL_DICT[config["General"]["model"]](num_classes=num_classes, input_channels=2 if USE_SEG_INPUT else 1).to(device)

    if args.start_epoch>0:
        checkpoint = torch.load(model_path.replace('best_model', 'latest_model'))
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"], weight_decay=1e-6)
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif model_path != "":
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer=None
    else:
        init_weights(model, init_type='kaiming')
        optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"], weight_decay=1e-6)
    return model, optimizer, calculate_itermediate