from shutil import copyfile
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
import datetime
import csv
import nibabel as nib
import json
import math
from PIL import Image
from utils.metrics import Task

from utils.voreen_vesselgraphextraction import extract_vessel_graph

class Visualizer():
    """
    The Visualizer takes care of all output related functionality.
    """
    def __init__(self, config: dict, continue_train=False, USE_SEG_INPUT=False) -> None:
        """
        Create a visualizer class with the given config that takes care of image and metric saving, as well as the interaction with tensorboard.
        """
        self.config = config
        self.save_to_disk: bool = config["Output"]["save_to_disk"]
        self.save_to_tensorboard: bool = config["Output"]["save_to_tensorboard"]

        if not os.path.isdir(config["Output"]["save_dir"]):
            os.mkdir(config["Output"]["save_dir"])

        self.track_record: list[dict[str, dict[str, float]]] = list()
        self.epochs = []
        self.log_file_path = None

        if continue_train:
            name = config["Output"]["save_dir"].split("/")[-1]
            self.save_dir = os.path.join(config["Output"]["save_dir"][:-len(name)], datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.mkdir(self.save_dir)
            self._copy_log_file(config["Output"]["save_dir"], self.save_dir)
            self._copy_best_checkpoint(config["Output"]["save_dir"], self.save_dir)
            with open(self.log_file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    d = dict()
                    if config["General"]["task"] == Task.GAN_VESSEL_SEGMENTATION:
                        d["loss"] = {k: float(v) for k,v in list(row.items())[1:-2]}
                        d["metric"] = {k: float(v) for k,v in list(row.items())[-2:]}
                    else:
                        d["loss"] = {k: float(v) for k,v in list(row.items())[1:3]}
                        d["metric"] = {k: float(v) for k,v in list(row.items())[3:]}
                    self.track_record.append(d)
                    self.epochs.append(int(row["epoch"]))
            if self.save_to_tensorboard:
                self.tb = SummaryWriter(log_dir=self.save_dir)
                for epoch,metric_groups in zip(self.epochs,self.track_record):
                    for title,record in metric_groups.items():
                        self.tb.add_scalars(title, record, epoch+1)
                        for k,v in record.items():
                            self.tb.add_scalar(k,v,epoch+1)
        else:
            if USE_SEG_INPUT:
                config["Output"]["save_dir"] = os.path.join(config["Output"]["save_dir"], "pre_ves_seg")
                if not os.path.isdir(config["Output"]["save_dir"]):
                    os.mkdir(config["Output"]["save_dir"])

            self.save_dir = os.path.join(config["Output"]["save_dir"], datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.mkdir(self.save_dir)
            if self.save_to_tensorboard:
                self.tb = SummaryWriter(log_dir=self.save_dir)
        
        config["Output"]["save_dir"] = self.save_dir
        config["Test"]["save_dir"] = os.path.join(self.save_dir, 'test/')
        config["Validation"]["save_dir"] = os.path.join(self.save_dir, 'val/')
        config["Test"]["model_path"] = os.path.join(self.save_dir, 'best_model.pth')
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def _prepare_log_file(self, record: dict[str, dict[str, float]]):
        titles = [title for v in record.values() for title in v]
        self.log_file_path = os.path.join(self.save_dir, 'metrics.csv')
        with open(self.log_file_path, 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", *titles])

    def _copy_log_file(self, old_dir, new_dir):
        old_log_file_path = os.path.join(old_dir, 'metrics.csv')
        self.log_file_path = os.path.join(new_dir, 'metrics.csv')
        copyfile(old_log_file_path,self.log_file_path)

    def _copy_best_checkpoint(self, old_dir, new_dir):
        if self.config["General"]["task"] == Task.GAN_VESSEL_SEGMENTATION:
            old_checkpoint = os.path.join(old_dir, 'latest_G_model.pth')
            new_checkpoint = os.path.join(new_dir, 'latest_G_model.pth')
            copyfile(old_checkpoint,new_checkpoint)

            old_checkpoint = os.path.join(old_dir, 'latest_D_model.pth')
            new_checkpoint = os.path.join(new_dir, 'latest_D_model.pth')
            copyfile(old_checkpoint,new_checkpoint)
            
            old_checkpoint = os.path.join(old_dir, 'latest_S_model.pth')
            new_checkpoint = os.path.join(new_dir, 'latest_S_model.pth')
            copyfile(old_checkpoint,new_checkpoint)

        else:
            old_best_checkpoint = os.path.join(old_dir, 'best_model.pth')
            new_best_checkpoint = os.path.join(new_dir, 'best_model.pth')
            copyfile(old_best_checkpoint,new_best_checkpoint)
            old_latest_checkpoint = os.path.join(old_dir, 'latest_model.pth')
            new_latest_checkpoint = os.path.join(new_dir, 'latest_model.pth')
            copyfile(old_latest_checkpoint,new_latest_checkpoint)

    def plot_losses_and_metrics(self, metric_groups: dict[str, dict[str, float]], epoch: int):
        """
        Plot the given losses and metrics.
        If save_to_disk is true, create a matpyplot figure.
        If save_to_tensorboard add the scalars to the current tensorboard

        Parameters:
            - metric_groups: A dictionary containing the loss and metric groups that are being plottet together in on graph.
            Each entry is a dictionary all metric labels and values of the cureent step.
            - epoch: The current epoch / step
        """
        self.track_record.append(dict())
        if self.log_file_path is None:
            self._prepare_log_file(metric_groups)

        for title, metrics in metric_groups.items():
            self.track_record[-1][title] = metrics
        self.epochs.append(epoch)

        with open(self.log_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, *[title for v in metric_groups.values() for title in v.values()]])

        if self.save_to_disk:
            loss_fig, loss_fig_axes = plt.subplots(1,len(metric_groups), figsize=(len(metric_groups)*6,5))
            if not isinstance(loss_fig_axes, list):
                loss_fig_axes = [loss_fig_axes]
            i=0
            for title, record in self.track_record[-1].items():
                data_y = [[v for v in data_t[title].values()] for data_t in self.track_record]

                ax: plt.Axes = loss_fig_axes[i]
                ax.clear()
                ax.set_title(title)
                ax.set_xlabel("epoch")
                ax.plot(self.epochs, data_y)
                ax.legend(list(record.keys()))

                i+=1
            plt.savefig(os.path.join(self.save_dir, 'loss.png'), bbox_inches='tight')
            plt.close()
        
        if self.save_to_tensorboard:
            for title,record in metric_groups.items():
                self.tb.add_scalars(title, record, epoch+1)
                for k,v in record.items():
                    self.tb.add_scalar(k,v,epoch+1)


    def plot_clf_sample(self, input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor, path: str, suffix: int = None):
        plot_clf_sample(
            save_dir=self.save_dir,
            input=input,
            pred=pred,
            truth=truth,
            path=path,
            suffix=suffix,
            save_to_tensorboard=self.save_to_tensorboard,
            tb=self.tb,
            save_to_disk=self.save_to_disk
        )

    def plot_sample(self,input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor = None, path: str='', suffix: str = None):
        """
        Create a 3x1 (or 2x1 if no truth tensor is supplied) grid from the given 2D image tensors and save the image with the given number as label.
        If save_to_disk is true, create a matpyplot figure.
        If save_to_tensorboard add the images to the current tensorboard
        """
        plot_sample(
            save_dir=self.save_dir,
            input=input,
            pred=pred,
            truth=truth,
            path=path,
            suffix=suffix,
            save_to_tensorboard=self.save_to_tensorboard,
            tb=self.tb,
            save_to_disk=self.save_to_disk)

    def _count_parameters(self, model: torch.nn.Module):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        s = str(table)
        s += f"\nTotal Trainable Params: {total_params}"
        return s

    def save_model_architecture(self, model: torch.nn.Module, batch):
        with open(os.path.join(self.save_dir, 'architecture.txt'), 'w+') as f:
            f.writelines(str(model))
            f.write("\n")
            f.writelines(self._count_parameters(model))
        if self.save_to_tensorboard:
            self.tb.add_graph(model, batch)

    def save_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, prefix: str, save_epoch=False):
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            os.path.join(self.save_dir, f"{prefix}_model.pth"),
        )
        if save_epoch:
            copyfile(os.path.join(self.save_dir, f"{prefix}_model.pth"), os.path.join(self.save_dir, f"{epoch}_{prefix}_model.pth"))

    def save_tune_checkpoint(path: str, d: dict):
        path = os.path.join(path, "checkpoint.pth")
        torch.save(
            d,
            path,
        )
        return path

    def log_model_params(self, model: torch.nn.Module, epoch: int):
        for name, weight in model.named_parameters():
            self.tb.add_histogram(name,weight, epoch)
            if not torch.isnan(weight.grad).any() and not torch.isinf(weight.grad).any():
                self.tb.add_histogram(f'{name}.grad',weight.grad, epoch)

    def save_hyperparams(self, params: dict, metrics):
        self.tb.add_hparams(params, metrics)

    def get_max_of_metric(self, metric_type: str, metric_name:str) -> tuple[float, int]:
        metric_values = [m[metric_type][metric_name] for m in self.track_record]
        return max(metric_values), np.argmax(metric_values)

    def plot_gan_seg_sample(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        fake_B_seg: torch.Tensor,
        real_B: torch.Tensor,
        idt_B: torch.Tensor,
        real_B_seg: torch.Tensor,
        epoch: int,
        path_A: str,
        path_B: str,
        save_epoch=False,
        full_size=True):
        
        images = {
            "real_A": real_A,
            "fake_B": fake_B,
            "fake_B_seg": fake_B_seg,
            "real_B": real_B,
            "idt_B": idt_B,
            "real_B_seg": real_B_seg
        }
        images = {k: (v.squeeze().detach().clip(0,1).cpu().numpy() * 255).astype(np.uint8) for k,v in images.items() if v is not None}

        name_A = path_A.split("/")[-1]
        name_B = path_B.split("/")[-1]

        inches = get_fig_size(real_A) / (1 if full_size else 2)
        fig, _ = plt.subplots(2, 3, figsize=(3*inches, 2*inches))
        plt.title(f"A: {name_A}, B: {name_B}")
        for i, (title, img) in enumerate(images.items()):
            fig.axes[i].imshow(img)#, cmap='Greys')
            fig.axes[i].set_title(title)
        path = os.path.join(self.save_dir, f'latest.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        if save_epoch:
            copyfile(path, os.path.join(self.save_dir, f'{epoch}.png'))

    def plot_cut_sample(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
        idt_B: torch.Tensor,
        epoch: int,
        path_A: str,
        path_B: str,
        save_epoch=False,
        full_size=True):
        
        images = {
            "real_A": real_A,
            "real_B": real_B,
            "fake_B": fake_B,
            "idt_B": idt_B
        }
        images = {k: (v.squeeze().detach().clip(0,1).cpu().numpy() * 255).astype(np.uint8) if v is not None else np.zeros((2,2), dtype=np.uint8) for k,v in images.items()}

        name_A = path_A.split("/")[-1]
        name_B = path_B.split("/")[-1]

        inches = get_fig_size(real_A) / (1 if full_size else 2)
        fig, _ = plt.subplots(2, 2, figsize=(2*inches, 2*inches))
        plt.title(f"A: {name_A}, B: {name_B}")
        for i, (title, img) in enumerate(images.items()):
            fig.axes[i].imshow(img, cmap='gray')
            fig.axes[i].set_title(title)
        path = os.path.join(self.save_dir, f'latest.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        if save_epoch:
            copyfile(path, os.path.join(self.save_dir, f'{epoch}.png'))

def plot_single_image(save_dir:str, input: torch.Tensor, name:str=None):
    Image.fromarray((input.squeeze().detach().cpu().numpy()*255).astype(np.uint8)).save(os.path.join(save_dir, name))

def plot_sample(
    save_dir: str,
    input: torch.Tensor,
    pred: torch.Tensor,
    truth: torch.Tensor=None,
    path: str = '',
    suffix:int=None,
    save_to_tensorboard=False,
    tb: SummaryWriter=None,
    save_to_disk=True,
    full_size=False):
    """
    Create a 3x1 (or 2x1 if no truth tensor is supplied) grid from the given 2D image tensors and save the image with the given number as label
    """
    input = input.squeeze().detach().cpu().numpy()
    if len(input.shape)==3:
        input=input[0]
    input = input - input.min()
    input = input / input.max()
    input = (input * 255).astype(np.uint8)
    
    if truth is not None:
        truth = truth.squeeze().detach().cpu().numpy()
        truth = (truth * 255).astype(np.uint8)

    pred = pred.squeeze().detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)

    name = path.split("/")[-1]
    n = 2 if truth is None else 3

    if save_to_tensorboard:
        # TODO use n
        if truth is not None:
            if (len(pred.shape)==3):
                input = np.tile(input[np.newaxis,:,:],[3,1,1])
                images = np.stack([input, pred, truth])
            else:
                images = np.expand_dims(np.stack([input, pred, truth]),1)
            label = f"{name} - Input, Pred, Truth"
        else:
            if (len(pred.shape)==3):
                input = np.tile(input[np.newaxis,:,:],[3,1,1])
                images = np.stack([input, pred])
            else:
                images = np.expand_dims(np.stack([input, pred]),1)
            label = "Input, Pred"
        tb.add_images(label, images)
    
    if save_to_disk:
        inches = get_fig_size(input) / (1 if full_size else 2)
        fig, _ = plt.subplots(1, n, figsize=(n*inches, inches))
        fig.axes[0].set_title(f"{name} - Input")
        if (input.shape[0]==3):
            input = np.moveaxis(input,0,-1)
        fig.axes[0].imshow(input)#, cmap='Greys')

        fig.axes[1].set_title("Prediction")
        if (pred.shape[0]==3):
            pred = np.moveaxis(pred,0,-1)
        fig.axes[1].imshow(pred)#, cmap='Greys')

        if truth is not None:
            fig.axes[2].set_title("Truth")
            if (truth.shape[0]==3):
                truth = np.moveaxis(truth,0,-1)
            fig.axes[2].imshow(truth)#, cmap='Greys')

        if suffix is not None:
            suffix = '_'+suffix
        else:
            suffix=''
        plt.savefig(os.path.join(save_dir, f'sample{suffix}.png'), bbox_inches='tight')
        plt.close()

def plot_clf_sample(
        save_dir: str,
        input: torch.Tensor,
        pred: torch.Tensor,
        truth: torch.Tensor,
        path: str,
        suffix: int = None,
        save_to_tensorboard=False,
        tb: SummaryWriter=None,
        save_to_disk: bool=True):
    input = input.squeeze(1).detach().cpu().numpy()
    input = input - input.min()
    input = input / input.max()
    input = (input * 255).astype(np.uint8)
    if save_to_disk or save_to_tensorboard:
        inches = get_fig_size(input)/2
        n = min(3,input.shape[0])
        fig = plt.subplots(1, n, figsize=(n*inches, inches))[0]
        for i in range(n):
            name = path[i].split("/")[-1]
            fig.axes[i].set_title(f"{name} - Pred: {np.round(pred[i].detach().cpu().numpy(),2)}, Real: {truth[i].detach().cpu().numpy()}")
            if len(input[i].shape)==3:
                fig.axes[i].imshow(input[i][-1])#, cmap='Greys')
            else:
                fig.axes[i].imshow(input[i])#, cmap='Greys')
        if suffix is not None:
            suffix = '_'+suffix
        else:
            suffix=''
        plt.savefig(os.path.join(save_dir, f'sample{suffix}.png'), bbox_inches='tight')
    if save_to_tensorboard:
        tb.add_figure('sample', fig)
    elif save_to_disk:
        plt.close()

def create_slim_3D_volume(img_2D_proj: torch.Tensor, save_dir: str, number: int = None):
    """
    Create a 3D nib file from a 2D image by adding a zero layers on each side of the input in a third dimension.
    """
    a = img_2D_proj.cpu().numpy().squeeze() * 255
    a = np.stack([np.zeros_like(a), a, np.zeros_like(a)], axis=-1)
    img_nii = nib.Nifti1Image(a.astype(np.uint8), np.eye(4))
    nib.save(img_nii, os.path.join(save_dir, f'sample{number}.nii'))

def extract_vessel_graph_features(img_2D_proj: torch.Tensor, save_dir: str, voreen_config: dict, number: int):
    a = img_2D_proj.cpu().numpy().squeeze() * 255
    a = np.stack([np.zeros_like(a), a, np.zeros_like(a)], axis=-1)
    img_nii = nib.Nifti1Image(a.astype(np.uint8), np.eye(4))
    
    if not os.path.exists(voreen_config["tempdir"]):
        os.mkdir(voreen_config["tempdir"])
    nii_path = os.path.join(voreen_config["tempdir"], f'sample{number}.nii')
    nib.save(img_nii, nii_path)
    return extract_vessel_graph(nii_path, 
        save_dir,
        voreen_config["tempdir"],
        voreen_config["cachedir"],
        voreen_config["bulge_size"],
        voreen_config["workspace_file"],
        voreen_config["voreen_tool_path"],
        number=str(number)
    )

def get_fig_size(input: torch.Tensor):
    return input.shape[-1]/96+2

def eukledian_dist(pos1: tuple[float], pos2: tuple[float]) -> float:
    dist = [(a - b)**2 for a, b in zip(pos1, pos2)]
    dist = math.sqrt(sum(dist))
    return dist

def graph_file_to_img(filepath: str, shape: tuple[int]):
    with open(filepath) as file:
        j = json.load(file)
    img = torch.zeros(shape)
    max = math.sqrt(0.5)

    for edge in j["graph"]["edges"]:
        if "skeletonVoxels"in edge:
            for voxel in edge["skeletonVoxels"]:
                x = voxel["pos"][0]
                y = voxel["pos"][1]

                v1 = (int(x), int(y))
                v2 = (int(x), int(y)+1)
                v3 = (int(x)+1, int(y))
                v4 = (int(x)+1, int(y)+1)

                for v in [v1,v2,v3,v4]:
                    d = eukledian_dist((v[0]+.5, v[1]+.5),(x,y))
                    intensity = d/max
                    img[v] = 0 if intensity<0.5 else 1
    for node in j["graph"]["nodes"]:
        # if "voxels_" in node:
        for voxel in node["voxels_"]:
            img[int(voxel[0])-1:int(voxel[0])+2, int(voxel[1])-1:int(voxel[1])+2] = 0.5
    return img

def save_prediction_csv(save_dir: str, predictions: list[list]):
    with open(os.path.join(save_dir, 'predictions.csv'), 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(["case","class", "P0", "P1", "P2"])
            for prediction in predictions:
                writer.writerow(prediction)
