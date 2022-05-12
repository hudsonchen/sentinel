from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import cv2 as cv
import colour
import pickle
from tqdm import tqdm
from preprocess_utils import *

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = False

plt.rcParams["axes.titlesize"] = 17
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 1.0
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['axes.grid'] = True

location_list = [(2300, 2200), (4400, 2400), (5500, 3900)]
# threshold = np.array([77., 130., 110.]).reshape([1, 1, 3])

finished = []
threshold_list = []
directory = "./images/"


def stage_one(img, date, visualize):
    threshold_array = np.load(f'{directory}threshold_center.npy')
    img_idx = np.zeros(img.shape[:-1])

    for x, y in location_list:
        threshold = img[x, y, :][None, :]
        threshold_array = np.concatenate((threshold_array, threshold), axis=0)

    for i in tqdm(range(len(threshold_array))):
        threshold = threshold_array[i].reshape([1, 1, 3])
        img_distance = ((threshold - img) ** 2).mean(-1)
        img_distance = img_distance.flatten()
        img_idx = img_idx.flatten()
        bottomk = int(len(img_distance) / 3 / len(threshold_array))
        distance_ind = np.argpartition(img_distance, bottomk)[:bottomk]
        img_idx[distance_ind] = 1

    img_idx = img_idx.reshape(img.shape[:-1])
    mask = 1.0 - img_idx
    np.save(f'{directory}{date}_mask.npy', mask)

    if visualize:
        fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(13, 10))
        axs[0].imshow(img / 255.)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title("Original Image")
        axs[1].imshow(img_idx, cmap='gray')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title("Water Image")
        axs[2].imshow(mask.astype(int), cmap='gray')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_title("Water Image (Black and White)")
        plt.show()

    all_river_color = img[img_idx.astype(bool)]
    rand_ind = np.random.permutation(len(all_river_color))[:100]
    all_river_color_to_save = all_river_color[rand_ind]
    threshold_array = np.concatenate((all_river_color_to_save, threshold_array), axis=0)
    center = k_means(threshold_array)
    if visualize:
        threed_color_visualize(center)
    np.save(f'{directory}threshold_center.npy', center)
    return mask


if __name__ == '__main__':
    for file in os.listdir(directory):
        try:
            date = file.split("L2A_")[1][:8]
        except:
            continue

        flag = True

        if ".bmp" not in file:
            flag = False
        if not flag:
            raise Exception('Already finished!')

        file_tag = file.split(".")[0]
        img = Image.open(directory + file)
        img = img.crop((6884, 0, 10980, 6800))
        img = np.array(img).astype(np.float32)
        stage_one(img, date, True)
        break
