from PIL import Image
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

location_list = [(2300, 2200), (4400, 2400), (5500, 3900)]
# threshold = np.array([77., 130., 110.]).reshape([1, 1, 3])

finished = []
threshold_list = []

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

    if not flag:
        raise Exception('Already finished!')

    file_tag = file.split(".")[0]
    img = Image.open(directory + file)
    img = img.crop((6884, 0, 10980, 6800))
    img = np.array(img).astype(np.float32)

    for x, y in location_list:
        threshold = img[x, y, :].reshape([1, 1, 3])
        threshold_list.append(threshold)

threshold_array = np.array(threshold_list).squeeze()
kmeans = KMeans(n_clusters=5, random_state=0).fit(threshold_array)
center = (kmeans.cluster_centers_).astype(int)

np.save('threshold_center.npy', center)

# %%
x = []
y = []
z = []
c = []

for i in range(len(center)):
    pix = center[i]
    newCol = (pix[0] / 255, pix[1] / 255, pix[2] / 255)
    x.append(pix[0])
    y.append(pix[1])
    z.append(pix[2])
    c.append(newCol)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=c)
ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")
ax.set_xlim([255, 0])
ax.set_ylim([0, 255])
ax.set_zlim([0, 255])
plt.show()
