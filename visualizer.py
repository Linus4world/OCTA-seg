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

from voreen_vesselgraphextraction import extract_vessel_graph

class Visualizer():
    """
    The Visualizer takes care of all output related functionality.
    """
    def __init__(self, config: dict, continue_train=False) -> None:
        """
        Create a visualizer class with the given config that takes care of image and metric saving, as well as the interaction with tensorboard.
        """
        self.config = config
        self.save_to_disk: bool = config["Output"]["save_to_disk"]
        self.save_to_tensorboard: bool = config["Output"]["save_to_tensorboard"]

        if not os.path.isdir(config["Output"]["save_dir"]):
            os.mkdir(config["Output"]["save_dir"])

        self.track_record: list[dict[str, dict[str, list[float]]]] = list()
        self.epochs = []
        self.log_file_path = None

        if continue_train:
            self.save_dir = os.path.join(config["Output"]["save_dir"][:-15], datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.mkdir(self.save_dir)
            self._copy_log_file(config["Output"]["save_dir"], self.save_dir)
            with open(self.log_file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    d = dict()
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
            self.save_dir = os.path.join(config["Output"]["save_dir"], datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.mkdir(self.save_dir)
            if self.save_to_tensorboard:
                self.tb = SummaryWriter(log_dir=self.save_dir)
        
        config["Output"]["save_dir"] = self.save_dir
        config["Test"]["save_dir"] = os.path.join(self.save_dir, 'test/')
        config["Test"]["model_path"] = os.path.join(self.save_dir, 'best_metric_model.pth')
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
            loss_fig, loss_fig_axes = plt.subplots(1,len(metric_groups), figsize=(12,5))
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

    def plot_clf_sample(self, input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor, number: int = None):
        input = input.squeeze(1).detach().cpu().numpy()
        input = (input * 255).astype(np.uint8)
        if self.save_to_disk or self.save_to_tensorboard:
            inches = get_fig_size(input)/2
            n = min(3,input.shape[0])
            fig = plt.subplots(1, n, figsize=(n*inches, inches))[0]
            for i in range(n):
                fig.axes[i].set_title(f"Pred: {np.round(pred[i].detach().cpu().numpy(),2)}, Real: {truth[i].detach().cpu().numpy()}")
                fig.axes[i].imshow(input[i])#, cmap='Greys')
            if number is not None:
                number = '_'+str(number)
            else:
                number=''
            plt.savefig(os.path.join(self.save_dir, f'sample{number}.png'), bbox_inches='tight')
        if self.save_to_tensorboard:
            self.tb.add_figure('sample', fig)
        elif self.save_to_disk:
            plt.close()

    def plot_sample(self,input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor = None, number: int = None):
        """
        Create a 3x1 (or 2x1 if no truth tensor is supplied) grid from the given 2D image tensors and save the image with the given number as label.
        If save_to_disk is true, create a matpyplot figure.
        If save_to_tensorboard add the images to the current tensorboard
        """
        input = input.squeeze().detach().cpu().numpy()
        input = (input * 255).astype(np.uint8)
        
        if truth is not None:
            truth = truth.squeeze().detach().cpu().numpy()
            truth = (truth * 255).astype(np.uint8)

        pred = pred.squeeze().detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)
        
        if self.save_to_disk:
            inches = get_fig_size(input)
            fig, _ = plt.subplots(1, 3, figsize=(3*inches, inches))
            fig.axes[0].set_title("Input")
            fig.axes[0].imshow(input)#, cmap='Greys')

            fig.axes[1].set_title("Prediction")
            fig.axes[1].imshow(pred)#, cmap='Greys')

            if truth is not None:
                fig.axes[2].set_title("Truth")
                fig.axes[2].imshow(truth)#, cmap='Greys')

            if number is not None:
                number = '_'+str(number)
            else:
                number=''
            plt.savefig(os.path.join(self.save_dir, f'sample{number}.png'), bbox_inches='tight')
            plt.close()

        if self.save_to_tensorboard:
            if truth is not None:
                images = np.expand_dims(np.stack([input, pred, truth]),1)
                label = "Input, Pred, Truth"
            else:
                images = np.expand_dims(np.stack([input, pred]),1)
                label = "Input, Pred"
            self.tb.add_images(label, images)

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

    def save_model(self, model: torch.nn.Module, prefix: str):
        torch.save(
            model.state_dict(),
            os.path.join(self.save_dir, f"{prefix}_model.pth"),
        )

    def log_model_params(self, model: torch.nn.Module, epoch: int):
        for name, weight in model.named_parameters():
            self.tb.add_histogram(name,weight, epoch)
            self.tb.add_histogram(f'{name}.grad',weight.grad, epoch)

    def save_hyperparams(self, params: dict, metrics):
        self.tb.add_hparams(params, metrics)

def plot_sample(save_dir: str, input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor=None, number:int=None):
    """
    Create a 3x1 (or 2x1 if no truth tensor is supplied) grid from the given 2D image tensors and save the image with the given number as label
    """
    input = input.squeeze().detach().cpu().numpy()
    input = (input * 255).astype(np.uint8)
        
    pred = pred.squeeze().detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)

    if truth is not None:
        truth = truth.squeeze().detach().cpu().numpy()
        truth = (truth * 255).astype(np.uint8)

    inches = get_fig_size(input)/2
    n = 2 if truth is None else 3
    fig, _ = plt.subplots(1, n, figsize=(n*inches, inches))
    plt.cla()
    fig.axes[0].set_title("Input")
    fig.axes[0].imshow(input)#, cmap='gray')

    fig.axes[1].set_title("Prediction")
    fig.axes[1].imshow(pred)#, cmap='Greys')

    if truth is not None:
        fig.axes[2].set_title("Graph")
        fig.axes[2].imshow(truth)#, cmap='Greys')

    if number is not None:
        number = '_'+str(number)
    else:
        number=''
    plt.savefig(os.path.join(save_dir, f'sample{number}.png'), bbox_inches='tight')
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