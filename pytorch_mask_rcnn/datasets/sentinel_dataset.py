from PIL import Image
import numpy as np
import os
import torch
import pickle
from torch.utils.data import Dataset
__all__ = ['Sentinel_Dataset']

mean = np.array([0.485, 0.456, 0.406])[:, None, None]
std = np.array([0.229, 0.224, 0.225])[:, None, None]

train_date = ["20211229", "20211204", "20211114", "20211005"]
test_date = ["20211219", "20200905"]


def load_sentinel(args):
    directory = args.data_dir
    dict_list = []
    for file in sorted(os.listdir(directory)):
        idx = file.split(".pickle")[0][-2:]
        date = file.split("L2A_")[1][:8]
        with open(directory + file, 'rb') as handle:
            dict_ = pickle.load(handle)
        img = dict_["images"]
        img_array = np.array(img).transpose((2, 0, 1)).astype(np.float32) / 255.
        img_array = (img_array - mean) / std
        dict_["images"] = img_array
        dict_.update(date=date, idx=idx)
        dict_list.append(dict_)
    return dict_list


class Sentinel_Dataset(Dataset):
    def __init__(self, args, train):
        self.data = []
        dict_all = load_sentinel(args)
        total_length = len(dict_all)
        if train:
            self.data = dict_all[:int(total_length * 0.75)]
        else:
            self.data = dict_all[int(total_length * 0.75):]
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def concate_visualize(dict_all, outputs):
    return


def construct_target(masks):
    masks = torch.tensor(masks.astype(np.float32)[None, :, :])
    boxes = torch.tensor([[0.0, 0.0, masks.shape[0], masks.shape[1]]])
    labels = torch.tensor([1])
    img_id = torch.tensor([-10]) # Put -10 to distinguish from coco dataset
    target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks)
    return target

