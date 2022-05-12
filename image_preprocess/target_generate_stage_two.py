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


directory = "./images/"
lake_loc = {'lake_one': {'lake_pos': (1300, 1000, 1300 + 1500, 1000 + 1800),
                         'special_pos': [[2500, 1900], [2300, 1600], [2000, 2200],
                                         [2000, 2000], [2000, 1500], [1700, 1600],
                                         [2300, 1200], [2300, 2200], [2000, 2400],
                                         [2300, 2400], [2500, 2400], [1500, 2200]]},
            'lake_two': {'lake_pos': (4000, 1800, 4000 + 1200, 1800 + 1200),
                         'special_pos': [[4400, 2200], [4400, 2500], [4400, 2800],
                                         [4700, 2200], [4200, 2400]]},
            'lake_three': {'lake_pos': (4800, 2500, 4800 + 1700, 2500 + 1590),
                           'special_pos': [[5500, 3900], [5500, 3600], [5800, 3600],
                                           [5800, 3300], [6100, 3000], [5100, 3800],
                                           [4900, 3800]]}}


def fine_tune(x1, x2, y1, y2, img, mask, special_pos, visualize):
    mask_one = mask[x1:x2, y1:y2]
    img_one = img[x1:x2, y1:y2, :]
    img_one_idx = 1.0 - mask_one

    epochs = 3
    if visualize:
        fig, axs = plt.subplots(1, epochs + 1, constrained_layout=True, figsize=(20, 5))
        axs[0].imshow(img_one / 255.)
        axs[0].set_xticks([])
        axs[0].set_yticks([])

    for j in range(epochs):
        river_color = img_one[(img_one_idx).astype(bool)]
        rand_ind = np.random.permutation(len(river_color))[:100]
        river_color = river_color[rand_ind]
        threshold_array = np.load('threshold_center.npy')
        threshold_array = np.concatenate((river_color, threshold_array), axis=0)
        center = k_means(threshold_array, n_clusters=10)
        # threed_color_visualize(center)

        for i in tqdm(range(len(center))):
            threshold = threshold_array[i].reshape([1, 1, 3])
            img_distance = ((threshold - img_one) ** 2).mean(-1)
            img_distance = img_distance.flatten()
            img_one_idx = img_one_idx.flatten()
            bottomk = int(len(img_distance) / len(threshold_array))
            distance_ind = np.argpartition(img_distance, bottomk)[:bottomk]
            img_one_idx[distance_ind] = 1

        for x, y in special_pos:
            threshold = img[x, y, :].reshape([1, 1, 3])
            img_distance = ((threshold - img_one) ** 2).mean(-1)
            img_distance = img_distance.flatten()
            img_one_idx = img_one_idx.flatten()
            bottomk = int(len(img_distance) / len(special_pos))
            distance_ind = np.argpartition(img_distance, bottomk)[:bottomk]
            img_one_idx[distance_ind] = 1

        img_one_idx = img_one_idx.reshape(img_one.shape[:-1])

        if visualize:
            axs[j + 1].imshow(1.0 - img_one_idx, cmap='gray')
            axs[j + 1].set_xticks([])
            axs[j + 1].set_yticks([])
    if visualize:
        plt.show()
    mask[x1:x2, y1:y2] = 1.0 - img_one_idx
    return mask


def stage_two(img, date, visualize):
    mask = np.load(f'{directory}{date}_mask.npy')
    for lake in ['lake_one', 'lake_two', 'lake_three']:
        x1, y1, x2, y2 = lake_loc[lake]['lake_pos']
        special_pos = lake_loc[lake]['special_pos']
        mask = fine_tune(x1, x2, y1, y2, img, mask, special_pos, visualize)
    np.save(f'{directory}{date}_mask.npy', mask)
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
        stage_two(img, date, True)
        break
