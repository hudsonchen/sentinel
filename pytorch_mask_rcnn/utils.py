import os
import re
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = False

plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1.0
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['axes.grid'] = True

__all__ = ["save_ckpt", "Meter", "get_acc", "get_iou", "visualize", "merge"]

mean = np.array([0.485, 0.456, 0.406])[None, :, None, None]
std = np.array([0.229, 0.224, 0.225])[:, None, None]


def get_acc(outputs, target):
    mask = torch.sigmoid(outputs) > 0.5
    return (target == mask).float().mean().item()


def get_iou(outputs, target):
    mask = torch.sigmoid(outputs) > 0.5
    intersection = torch.logical_and(target, mask).to(torch.float32)
    union = torch.logical_or(target, mask).to(torch.float32)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score.item()


def merge(data, mask, final_mask):
    x1, x2, y1, y2 = data['location']
    date = data['date']
    for j, d in enumerate(date):
        if d not in final_mask:
            final_mask[d] = np.zeros([6400, 4096])
        x1_, x2_, y1_, y2_ = x1[j], x2[j], y1[j], y2[j]
        final_mask[d][x1_:x2_, y1_:y2_] = mask[j, :].cpu().numpy()
    return final_mask

def visualize(args, data, outputs, train_test):
    fig, axs = plt.subplots(args.batch_size, 2, figsize=(5, 8))
    image = data["images"]
    image = (image * std + mean).numpy()
    for j in range(args.batch_size):
        axs[j, 0].imshow(image[j, :].transpose((1, 2, 0)))
        axs[j, 0].set_xticks([])
        axs[j, 0].set_yticks([])
        mask = (torch.sigmoid(outputs) > 0.5).float()
        axs[j, 1].imshow(mask[j, :].cpu().numpy(), cmap='gray')
        axs[j, 1].set_xticks([])
        axs[j, 1].set_yticks([])
    if train_test == 'train':
        plt.suptitle("Train Images visualize", fontweight="bold")
    elif train_test == 'test':
        plt.suptitle("Test Images visualize", fontweight="bold")
    else:
        pass
    plt.subplots_adjust(left=0.05,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.05,
                        hspace=0.05)
    # plt.show()
    return fig


def save_ckpt(model, optimizer, epochs, ckpt_path, **kwargs):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]  = optimizer.state_dict()
    checkpoint["epochs"] = epochs
        
    for k, v in kwargs.items():
        checkpoint[k] = v
        
    prefix, ext = os.path.splitext(ckpt_path)
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)
    torch.save(checkpoint, ckpt_path)
    
    
class TextArea:
    def __init__(self):
        self.buffer = []
    
    def write(self, s):
        self.buffer.append(s)
        
    def __str__(self):
        return "".join(self.buffer)

    def get_AP(self):
        result = {"bbox AP": 0.0, "mask AP": 0.0}
        
        txt = str(self)
        values = re.findall(r"(\d{3})\n", txt)
        if len(values) > 0:
            values = [int(v) / 10 for v in values]
            result = {"bbox AP": values[0], "mask AP": values[12]}
            
        return result
    
    
class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:sum={sum:.2f}, avg={avg:.4f}, count={count}"
        return fmtstr.format(**self.__dict__)
    

