from PIL import Image
import numpy as np
import os
import torch
__all__ = ['load_sentinel']


def load_sentinel(args):
    directory = args.data_dir
    img_list = []
    target_list = []
    for file in sorted(os.listdir(directory)):
        if ".bmp" in file:
            img = Image.open(directory + file)
            img_array = np.array(img).transpose((2, 0, 1)).astype(np.float32)
            img_list.append(img_array)
        elif ".npy" in file:
            masks = np.load(directory + file)
            target = construct_target(masks)
            target_list.append(target)
        else:
            pass
    return torch.tensor(np.array(img_list) / 255.), target_list


def construct_target(masks):
    masks = torch.tensor(masks.astype(np.float32)[None, :, :])
    boxes = torch.tensor([[0.0, 0.0, masks.shape[0], masks.shape[1]]])
    labels = torch.tensor([1])
    img_id = torch.tensor([-10]) # Put -10 to distinguish from coco dataset
    target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks)
    return target

