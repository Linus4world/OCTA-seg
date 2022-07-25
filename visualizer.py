import matplotlib.pyplot as plt
import os
import torch
import numpy as np
# from monai.visualize import plot_2d_or_3d_image

loss_fig = plt.figure("train", (12, 6))
sample_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

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

def plot_sample(input: torch.Tensor, pred: torch.Tensor, truth: torch.Tensor, save_dir: str):
    input = input.squeeze().detach().cpu().numpy()
    input = (input * 255).astype(np.uint8)
    plt.figure(sample_fig.number)
    plt.cla()
    ax1.set_title("Input")
    ax1.imshow(input)#, cmap='Greys')

    pred = pred.squeeze().detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)
    plt.cla()
    ax2.set_title("Prediction")
    ax2.imshow(pred)#, cmap='Greys')

    truth = truth.squeeze().detach().cpu().numpy()
    truth = (truth * 255).astype(np.uint8)
    ax3.set_title("Truth")
    ax3.imshow(truth)#, cmap='Greys')
    plt.savefig(os.path.join(save_dir, 'sample.png'), bbox_inches='tight')
