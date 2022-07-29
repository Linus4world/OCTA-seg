import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from prettytable import PrettyTable
# from monai.visualize import plot_2d_or_3d_image

loss_fig = plt.figure("train", (12, 6))
sample_fig, axes = plt.subplots(1, 3, figsize=(54, 18))
sample_fig_small, axes = plt.subplots(1, 2, figsize=(36, 18))

def plot_losses_and_metrics(epoch_loss_values: list[float], metric_values: list[float], val_interval: int, save_dir: str):
    plt.figure(loss_fig.number)
    plt.cla()
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig(os.path.join(save_dir, 'loss.png'), bbox_inches='tight')

def plot_sample( save_dir: str, input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor = None, number: int = None, save=True):
    input = input.squeeze().detach().cpu().numpy()
    input = (input * 255).astype(np.uint8)
    fig = plt.figure(sample_fig.number if truth is not None else sample_fig_small.number)
    fig.axes[0].set_title("Input")
    fig.axes[0].imshow(input)#, cmap='Greys')

    pred = pred.squeeze().detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    fig.axes[1].set_title("Prediction")
    fig.axes[1].imshow(pred)#, cmap='Greys')

    if truth is not None:
        truth = truth.squeeze().detach().cpu().numpy()
        truth = (truth * 255).astype(np.uint8)
        fig.axes[2].set_title("Truth")
        fig.axes[2].imshow(truth)#, cmap='Greys')

    if number is not None:
        number = '_'+str(number)
    else:
        number=''
    if save:
        plt.savefig(os.path.join(save_dir, f'sample{number}.png'), bbox_inches='tight')
    else:
        plt.show()

def count_parameters(model):
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

def save_model_architecture(model, save_dir: str):
    with open(os.path.join(save_dir, 'architecture.txt'), 'w+') as f:
        f.writelines(str(model))
        f.write("\n")
        f.writelines(count_parameters(model))
