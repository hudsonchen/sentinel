from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from target_generate_stage_one import *
from target_generate_stage_two import *

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

finished = []


def init():
    for file in os.listdir(directory):
        try:
            date = file.split("L2A_")[1][:8]
        except:
            continue

        flag = True

        if ".bmp" not in file:
            flag = False
        for finished_tag in finished:
            if finished_tag in file:
                flag = False
        if not flag:
            raise Exception('Already finished!')

        img = Image.open(directory + file)
        img = img.crop((6884, 0, 10980, 6800))
        img = np.array(img).astype(np.float32)

        for x, y in location_list:
            threshold = img[x, y, :].reshape([1, 1, 3])
            threshold_list.append(threshold)

    center = k_means(np.array(threshold_list))
    np.save(f'{directory}threshold_center_init.npy', center)
    np.save(f'{directory}threshold_center.npy', center)


def generate_mask():
    for file in os.listdir(directory):
        try:
            date = file.split("L2A_")[1][:8]
        except:
            continue

        flag = True

        if ".bmp" not in file:
            flag = False
        for finished_tag in finished:
            if finished_tag in file:
                flag = False

        if flag:
            file_tag = file.split(".")[0]
            img = Image.open(directory + file)
            img = img.crop((6884, 0, 10980, 6800))
            img = np.array(img).astype(np.float32)

            mask = stage_one(img, date, verbose_visualize)
            mask = stage_two(mask, img, date, verbose_visualize)

            if visualize:
                fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(13, 6))
                axs[0].imshow(img / 255.)
                axs[0].set_xticks([])
                axs[0].set_yticks([])
                axs[0].set_title("Original Image")
                axs[1].imshow(mask, cmap='gray')
                axs[1].set_xticks([])
                axs[1].set_yticks([])
                axs[1].set_title("Target")
                plt.show()


def cut_and_crop():
    for file in os.listdir(directory):
        try:
            date = file.split("L2A_")[1][:8]
        except:
            continue

        flag = True

        if ".bmp" not in file:
            flag = False

        # if date not in ["20211219", "20200905"]:
        #     flag = False

        if flag:
            file_tag = file.split(".")[0]
            print(date)
            img = Image.open(directory + file)
            img = img.crop((6884, 0, 10980, 6400))
            img = np.array(img).astype(np.float32)
            mask = np.load(f'{directory}{date}_mask.npy')

            image_x = 800
            image_y = 1024
            num = 1
            step_x = int(image_x / num)
            step_y = int(image_y / num)
            len_x = int((img.shape[0] - image_x) / step_x) + 1
            len_y = int((img.shape[1] - image_y) / step_y) + 1

            for i in range(len_x):
                for j in range(len_y):
                    idx = i * len_y + j
                    idx = str(idx) if idx > 10 else "0" + str(idx)
                    dict_to_save = {
                        "images": img[i * step_x:i * step_x + image_x, j * step_y:j * step_y + image_y, :].astype(
                            np.float32),
                        "target": mask[i * step_x:i * step_x + image_x, j * step_y:j * step_y + image_y].astype(np.float32),
                        "location": (i * step_x, i * step_x + image_x, j * step_y, j * step_y + image_y)}
                    with open(save_directory + file_tag + str(idx) + ".pickle", 'wb') as handle:
                        pickle.dump(dict_to_save, handle)

            ss = 0

# img_thres = img_np * mask[:, :, None]
# ratio = mask.sum() / mask.shape[0] / mask.shape[1]
# threshold = (img_thres.mean(0).mean(0) / ratio).reshape([1, 1, 3])


# fig, axs = plt.subplots(3, 3, constrained_layout=True, figsize=(13, 10))
# # idx_list = ["00", ""]
#
# for idx in range(len(axs.flatten())):
#     ax = axs.flatten()[idx]
#     idx = str(idx) if idx > 10 else "0" + str(idx)
#     with open(save_directoy + file_tag + str(idx) + ".pickle", 'rb') as handle:
#         dict_all = pickle.load(handle)
#     image_npy = dict_all['images']
#     ax.imshow(image_npy / 255.)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.subplots_adjust(left=0.05,
#                     bottom=0.05,
#                     right=0.95,
#                     top=0.95,
#                     wspace=0.1,
#                     hspace=0.1)
# plt.tight_layout()
# plt.savefig(plot_directory + date + "__" + "nine_subplots.pdf", bbox_inches='tight')
# plt.show()


if __name__ == '__main__':
    directory = "./images/"
    save_directory = '../data/sentinel_images/'
    plot_directory = '../plot/'
    visualize = True
    verbose_visualize = False
    init()
    generate_mask()
    cut_and_crop()


