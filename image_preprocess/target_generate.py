from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2 as cv
import colour
import pickle
from preprocess_utils import *
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
threshold = np.array([77., 130., 110.]).reshape([1, 1, 3])

finished = []


directory = "/Users/hudsonchen/hudson/research/thu_image_seg/sentinel_figures/sentinel_lakes/images_bmp/"
plot_directory = "/Users/hudsonchen/hudson/research/thu_image_seg/plots/sentinel_plots/"

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
        img = img.crop((6884,0,10980,6800))
        img_np = np.array(img).astype(np.float32)

        plt.figure()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title("Original Image")
        plt.show()
        

        for i in range(len_x):
            for j in range(len_y):
                idx = i * len_y + j
                idx = str(idx) if idx > 10 else "0" + str(idx)
                dict_to_save = {"images": img_np[i*step_x:i*step_x+image_x, j*step_y:j*step_y+image_y, :].astype(np.float32),
                     "target": mask[i*step_x:i*step_x+image_x, j*step_y:j*step_y+image_y].astype(np.float32)}
                with open(save_directoy + file_tag + str(idx) + ".pickle", 'wb') as handle:
                    pickle.dump(dict_to_save, handle)
    
# img_thres = img_np * mask[:, :, None]
# ratio = mask.sum() / mask.shape[0] / mask.shape[1]
# threshold = (img_thres.mean(0).mean(0) / ratio).reshape([1, 1, 3])


fig, axs = plt.subplots(3,3, constrained_layout = True, figsize=(13, 10))
# idx_list = ["00", ""]

for idx in range(len(axs.flatten())):
    ax = axs.flatten()[idx]
    idx = str(idx) if idx > 10 else "0" + str(idx)
    with open(save_directoy + file_tag + str(idx) + ".pickle", 'rb') as handle:
        dict_all = pickle.load(handle)
    image_npy = dict_all['images']
    ax.imshow(image_npy / 255.)
    ax.set_xticks([])
    ax.set_yticks([])
plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.1)
plt.tight_layout()
plt.savefig(plot_directory + date + "__" + "nine_subplots.pdf", bbox_inches='tight')
plt.show()

