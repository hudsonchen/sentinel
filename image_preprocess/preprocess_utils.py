import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.patches as patches


def threed_color_visualize(color):
    x = []
    y = []
    z = []
    c = []

    for i in range(len(color)):
        pix = color[i]
        newCol = (pix[0] / 255, pix[1] / 255, pix[2] / 255)
        x.append(pix[0])
        y.append(pix[1])
        z.append(pix[2])
        c.append(newCol)

    fig = plt.figure(constrained_layout=True)
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c=c)
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_xlim([255, 0])
    ax.set_ylim([0, 255])
    ax.set_zlim([0, 255])
    plt.show()
    return


def k_means(threshold, n_clusters=10):
    threshold = threshold.squeeze()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(threshold)
    center = kmeans.cluster_centers_.astype(int)
    return center


def visulize_points_rect(img, lake_loc):
    lake_name = ['lake_one', 'lake_two', 'lake_three']
    color = ['red', 'orange', 'pink']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img / 255.)
    plt.xticks([])
    plt.yticks([])

    for i, lake in enumerate(lake_name):
        x1, y1, x2, y2 = lake_loc[lake]['lake_pos']
        for special_pos in lake_loc[lake]['special_pos']:
            plt.scatter(special_pos[0], special_pos[1], s=5, c=color[i], marker='o')
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color[i], facecolor='none')
        ax.add_patch(rect)
    plt.show()
    return fig


def visualize_lake_location(img):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img / 255.)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(2200, 2300, s=5, c='red', marker='o')
    plt.scatter(1900, 2300, s=5, c='red', marker='o')
    plt.scatter(1600, 2300, s=5, c='red', marker='o')

    plt.scatter(2200, 2000, s=5, c='red', marker='o')
    plt.scatter(1900, 2000, s=5, c='red', marker='o')
    plt.scatter(1600, 2000, s=5, c='red', marker='o')

    plt.scatter(1600, 1700, s=5, c='red', marker='o')

    plt.scatter(1300, 2300, s=5, c='red', marker='o')

    plt.scatter(2200, 4400, s=5, c='orange', marker='o')
    plt.scatter(2500, 4400, s=5, c='orange', marker='o')
    plt.scatter(2800, 4400, s=5, c='orange', marker='o')

    plt.scatter(2200, 4700, s=5, c='orange', marker='o')

    plt.scatter(3900, 5500, s=5, c='pink', marker='o')
    plt.scatter(3600, 5500, s=5, c='pink', marker='o')

    plt.scatter(3600, 5800, s=5, c='pink', marker='o')
    plt.scatter(3300, 5800, s=5, c='pink', marker='o')
    plt.scatter(3000, 6300, s=5, c='pink', marker='o')

    # Create a Rectangle patch
    rect = patches.Rectangle((1000, 1300), 1800, 1500, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((1800, 4000), 1200, 1200, linewidth=1, edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((2500, 4800), 1590, 1700, linewidth=1, edgecolor='pink', facecolor='none')
    ax.add_patch(rect)
    plt.show()
    return
