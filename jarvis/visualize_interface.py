import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from jarvis.utils.utils import CLIColors
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
import jarvis.prediction.prediction_utils as utils
import jarvis.visualization.visualization_utils as viz_utils


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.4*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def	visualize_2D_sample(dataset, mode, img_idx, skeleton_preset):
    sample = dataset.__getitem__(img_idx)
    img = (sample[0]*dataset.cfg.DATASET.STD+dataset.cfg.DATASET.MEAN)*255
    heatmaps = sample[1]
    keypoints = sample[2]
    img = img - np.min(img)
    img = img/np.max(img)*255
    img = cv2.resize(img, None, fx=3, fy = 3)
    if mode == 'CenterDetect':
        if keypoints[0,0] + keypoints[0,1] != 0:
            img = cv2.circle(img, (int(keypoints[0,0]*3),
                        int(keypoints[0,1]*3)), 4, (255,0,0), 6)
    else:
        colors = []
        line_idxs = []
        if skeleton_preset != "None":
            colors, line_idxs = utils.get_colors_and_lines(skeleton_preset)
            if len(colors) < dataset.cfg.KEYPOINTDETECT.NUM_JOINTS:
                print (f'{CLIColors.FAIL}Number of keypoints in dataset does '
                            f'not match number of keypoints in skeleton '
                            f'preset! Using default colormap.{CLIColors.ENDC}')
                colors = []
                line_idxs = []
        if len(colors) == 0:
            cmap = matplotlib.cm.get_cmap('jet')
            for i in range(dataset.cfg.KEYPOINTDETECT.NUM_JOINTS):
                colors.append(((np.array(
                cmap(float(i)/dataset.cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                            255).astype(int)[:3]).tolist())
        keypoints = keypoints.reshape(-1,3)
        for i,keypoint in enumerate(keypoints):
            if keypoint[0] + keypoint[1] != 0:
                img = cv2.circle(img, (int(keypoint[0]*3), int(keypoint[1]*3)),
                            4, colors[i], 6)
        for line in line_idxs:
            if (keypoints[line[0]][0] + keypoints[line[0]][1] != 0
                        and keypoints[line[1]][0] + keypoints[line[1]][1] != 0):
                cv2.line(img, (int(keypoints[line[0]][0]*3),
                            int(keypoints[line[0]][1]*3)),
                            (int(keypoints[line[1]][0]*3),
                            int(keypoints[line[1]][1]*3)),
                    colors[line[1]], 1)
    return img/255


def visualize_3D_sample(dataset, img_idx, skeleton_preset, azim = 0, elev = 0):
    colors = []
    line_idxs = []
    if skeleton_preset != "None":
        colors, line_idxs = utils.get_colors_and_lines(skeleton_preset)
        if len(colors) < dataset.cfg.KEYPOINTDETECT.NUM_JOINTS:
            print (f'{CLIColors.FAIL}Number of keypoints in dataset does '
                        f'not match number of keypoints in skeleton '
                        f'preset! Using default colormap.{CLIColors.ENDC}')
            colors = []
            line_idxs = []
    if len(colors) == 0:
        cmap = matplotlib.cm.get_cmap('jet')
        for i in range(dataset.cfg.KEYPOINTDETECT.NUM_JOINTS):
            colors.append(((np.array(
            cmap(float(i)/dataset.cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                        255).astype(int)[:3]).tolist())
    sample = dataset.__getitem__(img_idx)
    images = sample[0]
    keypoints3D = sample[1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.margins(0)
    ax.azim = azim
    ax.elev = elev
    for i, point in enumerate(keypoints3D):
        ax.scatter(point[0], point[1], point[2],
                    color = tuple(np.array(colors[i])/255.))
    for line in line_idxs:
        ax.plot([keypoints3D[line[0]][0], keypoints3D[line[1]][0]],
                  [keypoints3D[line[0]][1], keypoints3D[line[1]][1]],
                  [keypoints3D[line[0]][2], keypoints3D[line[1]][2]],
                  color = tuple(np.array(colors[line[1]])/255.))
    set_axes_equal(ax)
    ax.autoscale_view('tight')
    return fig
