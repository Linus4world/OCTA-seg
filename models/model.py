from typing import Union, Literal
import torch
from models.cut import CUT
from models.gan_seg_model import GanSegModel
from utils.metrics import Task

from models.networks import MODEL_DICT, init_weights


def define_model(config: dict, phase: Literal["train", "val", "test"]) -> tuple[torch.nn.Module, Union[torch.optim.Optimizer, tuple[torch.optim.Optimizer]]]:
    # Model
    num_layers = config["General"]["num_layers"]
    kernel_size = config["General"]["kernel_size"]
    device = torch.device(config["General"]["device"])
    task: Task = config["General"]["task"]
    num_classes=config["Data"]["num_classes"]
    model_name: str = config["General"]["model"]
    if task == Task.VESSEL_SEGMENTATION:
        model = MODEL_DICT[model_name](
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
        model = MODEL_DICT[model_name](
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
        segmentor = None
        discriminator = None
        generator = None
        if phase != "test" or config["Test"]["inference"] == "S":
            segmentor=MODEL_DICT[config["General"]["model_s"]](
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=(3, *[kernel_size]*num_layers,3),
                strides=(1,*[2]*num_layers,1),
                upsample_kernel_size=(1,*[2]*num_layers,1)
            ).to(device)
        if phase != "test" or config["Test"]["inference"] == "G":
            generator = MODEL_DICT[config["General"]["model_g"]]().to(device)
        if phase != "test":
            discriminator=MODEL_DICT[config["General"]["model_d"]]().to(device)

        model = GanSegModel(
            generator,
            discriminator,
            segmentor,
            compute_identity=config["Train"]["compute_identity"],
            compute_identity_seg=config["Train"]["compute_identity_seg"]
        )
    elif task == Task.CONSTRASTIVE_UNPAIRED_TRANSLATION:
        model = CUT(
            generator=MODEL_DICT[config["General"]["model_g"]]().to(device),
            discriminator=MODEL_DICT[config["General"]["model_d"]]().to(device),
            segmentor=MODEL_DICT[config["General"]["model_s"]](
                device=device
            ).to(device),
            compute_identity=config["Train"]["compute_identity"],
            compute_identity_seg=config["Train"]["compute_identity_seg"],
            nce_layers=config["Train"]["nce_layers"]
        )
    return model

def initialize_model_and_optimizer(model: torch.nn.Module, config: dict, args, load_best=False, phase="train") -> Union[torch.optim.Optimizer, tuple[torch.optim.Optimizer]]:
    task = config["General"]["task"]
    model_path: str = config["Test"]["model_path"]
    model_name: str = config["General"]["model"]
    if task == Task.GAN_VESSEL_SEGMENTATION or task == Task.CONSTRASTIVE_UNPAIRED_TRANSLATION:
        model: GanSegModel = model
        if phase == "train":
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
            elif task == Task.GAN_VESSEL_SEGMENTATION: 
                for m, m_name in zip([model.generator, model.discriminator, model.segmentor], [config["General"]["model_g"], config["General"]["model_d"], config["General"]["model_s"]]):
                    activation = 'relu' if m_name.lower().startswith("resnet") else 'leaky_relu'
                    init_weights(m, init_type='kaiming', nonlinearity=activation)
            else:
                for m, m_name in zip([model.generator, model.discriminator], [config["General"]["model_g"], config["General"]["model_d"]]):
                    activation = 'relu' if (m_name.lower().startswith("resnet") or m_name.lower().startswith("patch")) else 'leaky_relu'
                    init_weights(m, init_type='kaiming', nonlinearity=activation)

            return (optimizer_G,optimizer_D,optimizer_S)
        elif task == Task.CONSTRASTIVE_UNPAIRED_TRANSLATION:
            epoch_prefix = f"{args.epoch}_" if args.epoch is not None else "" 
            checkpoint_G = torch.load(model_path.replace('best_model', epoch_prefix+'latest_G_model'))
            model.generator.load_state_dict(checkpoint_G['model'])
        else:
            epoch_prefix = f"{args.epoch}_" if args.epoch is not None else "" 
            checkpoint = torch.load(model_path.replace('best_model', epoch_prefix+f'latest_{config["Test"]["inference"]}_model'))
            if config["Test"]["inference"] == "S":
                model.segmentor.load_state_dict(checkpoint['model'])
            elif config["Test"]["inference"] == "G":
                model.generator.load_state_dict(checkpoint['model'])
            else: raise NotImplementedError
        return None

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
    return optimizer
