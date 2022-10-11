from typing import Union
import torch
from models.gan_seg_model import GanSegModel
from utils.metrics import Task

from models.networks import MODEL_DICT, init_weights


def initialize_model(config: dict, args, load_best=False) -> tuple[torch.nn.Module, Union[torch.optim.Optimizer, tuple[torch.optim.Optimizer]]]:
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
        if config["Data"]["enhance_vessels"]:
            input_channels=1
        else:
            input_channels=sum([True, config["Data"]["use_segmentation"], config["Data"]["use_background"]])
        model = MODEL_DICT[config["General"]["model"]](
            spatial_dims=2,
            in_channels=sum([True, config["Data"]["use_segmentation"], config["Data"]["use_background"]]),
            out_channels=num_classes,
            kernel_size=(3, *[kernel_size]*num_layers,3),
            strides=(1,*[2]*num_layers,1),
            upsample_kernel_size=(1,*[2]*num_layers,1),
        ).to(device)
        # Use pretrained
        if task == Task.AREA_SEGMENTATION and "pretrained_model" in config["Train"]:
            checkpoint = torch.load(config["Train"]["pretrained_model"])
            if 'model' in checkpoint:
                pretrained_dict = {k: v for k, v in checkpoint['model'].items() if
                                    (k in model.state_dict().keys()) and (model.state_dict()[k].shape == checkpoint['model'][k].shape)}
            else:
                # filter unnecessary keys
                pretrained_dict = {k: v for k, v in checkpoint.items() if
                                    (k in model.state_dict().keys()) and (model.state_dict()[k].shape == checkpoint[k].shape)}
            model.load_state_dict(pretrained_dict, strict=False)
    elif task == Task.IMAGE_QUALITY_CLASSIFICATION or task == Task.RETINOPATHY_CLASSIFICATION:
        if config["Data"]["enhance_vessels"]:
            input_channels=1
        else:
            input_channels=sum([True, config["Data"]["use_segmentation"], config["Data"]["use_background"]])
        model = MODEL_DICT[model_name](num_classes=num_classes, input_channels=input_channels).to(device)
    elif task == Task.GAN_VESSEL_SEGMENTATION:
        model = GanSegModel(
            generator=MODEL_DICT[config["General"]["model_g"]]().to(device),
            discriminator=MODEL_DICT[config["General"]["model_d"]]().to(device),
            segmentor=MODEL_DICT[config["General"]["model_s"]](
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=(3, *[kernel_size]*num_layers,3),
                strides=(1,*[2]*num_layers,1),
                upsample_kernel_size=(1,*[2]*num_layers,1)
            ).to(device),
            compute_identity = config["Train"]["compute_identity"],
            compute_identity_seg=config["Train"]["compute_identity_seg"]
        )
        for m, m_name in zip([model.generator, model.discriminator, model.segmentor], [config["General"]["model_g"], config["General"]["model_d"], config["General"]["model_s"]]):
            activation = 'relu' if m_name.lower().startswith("resnet") else 'leaky_relu'
            init_weights(m, init_type='kaiming', nonlinearity=activation)
        optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=config["Train"]["lr"], betas=(0.5 , 0.999))
        optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=config["Train"]["lr"], betas=(0.5 , 0.999))
        optimizer_S = torch.optim.Adam(model.segmentor.parameters(), lr=config["Train"]["lr"])
        if hasattr(args, "start_epoch") and args.start_epoch>0:
            checkpoint_G = torch.load(model_path.replace('best_model', 'latest_G_model'))
            model.generator.load_state_dict(checkpoint_G['model'])
            optimizer_G.load_state_dict(checkpoint_G['optimizer'])

            checkpoint_D = torch.load(model_path.replace('best_model', 'latest_D_model'))
            model.discriminator.load_state_dict(checkpoint_D['model'])
            optimizer_D.load_state_dict(checkpoint_D['optimizer'])

            checkpoint_S = torch.load(model_path.replace('best_model', 'latest_S_model'))
            model.segmentor.load_state_dict(checkpoint_S['model'])
            optimizer_S.load_state_dict(checkpoint_S['optimizer'])

        return model, (optimizer_G,optimizer_D,optimizer_S)
    else:
        raise NotImplementedError

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
        activation = 'relu' if model_name.lower().startswith("resnet") else 'leaky_relu'
        init_weights(model, init_type='kaiming', nonlinearity=activation)
        optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"], weight_decay=1e-6)
    return model, optimizer