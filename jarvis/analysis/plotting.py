"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from jarvis.config.project_manager import ProjectManager



def plot_error_histogram(path, additional_data = {}, cutoff = -1,
            interactive = True):
    gt_path = os.path.join(path, 'points_GroundTruth.csv')
    net_path = os.path.join(path, 'points_HybridNet.csv')

    sns.set_theme()
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context("paper", font_scale=1.25)

    pointsGT = np.genfromtxt(gt_path, delimiter=',')
    pointsGT = pointsGT.reshape(-1,int(pointsGT.shape[1]/3), 3)
    pointsNet = np.genfromtxt(net_path, delimiter=',')
    pointsNet = pointsNet.reshape(-1,int(pointsNet.shape[1]/3), 3)

    pointsList = [pointsNet]
    labels = ["JARVIS"]

    for preds in additional_data:
        labels += [preds]
        points = np.genfromtxt(additional_data[preds], delimiter=',')
        points = points.reshape(-1,int(points.shape[1]/3), 3)
        pointsList += [points]

    f, (ax_hist, ax_box) = plt.subplots(2, sharex=True,
                gridspec_kw= {"height_ratios": (1, 0.2)},
                figsize=(6.92913,6.92913 / 1.618))
    plt.suptitle("Euclidean Distance to Ground Truth across all joints")
    distances_l = {}
    for i,points in enumerate(pointsList):
        distances = np.sqrt(np.sum((points-pointsGT)**2, axis = 2))
        mask = np.sum(pointsGT,axis = 2)
        distances = distances[mask != 0]

        if cutoff != -1:
            distances[distances>cutoff] = cutoff
        distances_l[labels[i]] = (distances.reshape(-1))
    distances_pd = pd.DataFrame(distances_l)

    sns.boxplot(data = distances_pd, fliersize = 0, ax=ax_box, orient="h")
    sns.histplot(data = distances_pd, ax = ax_hist, element="step",alpha=0.1)
    labels.reverse()
    for i in range(len(labels)):
        labels[i] = labels[i] + f" ({np.median(distances_l[labels[i]]):.2f} mm)"
    ax_hist.legend(labels=labels, frameon=False)


    plt.xlabel('Deviation from manual annotations [mm]')
    if cutoff != -1:
        if cutoff < 15:
            step = 2
        else:
            step = 5
        plt.xlim(0,cutoff+0.1)
        x_labels = [str(i) for i in range(0,cutoff, step)] + [f'>{cutoff}']
        plt.xticks(list(step * np.arange(len(x_labels)-1))+[cutoff])
        ax_box.set_xticklabels(x_labels)
    plt.savefig(os.path.join(path, "error_histogram.png"))
    if interactive:
        plt.show()
    return f


def plot_error_per_keypoint(path, project_name, interactive = True):
    sns.set_theme()
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context("paper", font_scale=1.25)

    projectManager = ProjectManager()
    projectManager.load(project_name)
    cfg = projectManager.cfg

    fig = plt.figure()

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    plt.ylabel("Mean Deviation from manual annotations [mm]")
    plt.suptitle("Euclidean Distance to Ground Truth per Joint")

    gt_path = os.path.join(path, 'points_GroundTruth.csv')
    net_path = os.path.join(path, 'points_HybridNet.csv')

    sns.set_theme()
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context("paper", font_scale=1.25)

    pointsGT = np.genfromtxt(gt_path, delimiter=',')
    pointsGT = pointsGT.reshape(-1,int(pointsGT.shape[1]/3), 3)
    pointsNet = np.genfromtxt(net_path, delimiter=',')
    pointsNet = pointsNet.reshape(-1,int(pointsNet.shape[1]/3), 3)
    number_joints = pointsNet.shape[1]

    distances = np.sqrt(np.sum((pointsNet-pointsGT)**2, axis = 2))
    mask = np.sum(pointsGT,axis = 2)
    mask[mask == 0] = 1
    mask[mask != 1] = 0
    distances = np.ma.array(distances, mask=mask)
    joints_means = np.ma.mean(distances, axis = 0)

    barWidth = 0.8
    joints = np.arange(number_joints)
    joint_labels = [cfg.KEYPOINT_NAMES[i] for i in range(number_joints)]

    cmap = plt.cm.get_cmap('jet')
    for i in range(0,len(joints)):
        plt.bar(joints[i], joints_means[i],
                    width = barWidth, color = cmap(i*(1/number_joints)))

    plt.xticks([r + 0.1 for r in range(len(joints))],
                joint_labels, rotation=90)
    plt.savefig(os.path.join(path, "error_per_joint.png"))
    if interactive:
        plt.show()
    return fig


def plot_error_histogram_per_keypoint(path, project_name, cutoff = -1,
            interactive = True):
    sns.set_theme()
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context("paper", font_scale=1.25)

    projectManager = ProjectManager()
    projectManager.load(project_name)
    cfg = projectManager.cfg

    os.makedirs(os.path.join(path, "keypoint_histograms"), exist_ok = True)

    num_keypoints = len(cfg.KEYPOINT_NAMES)
    keypoint_grid_height = int(np.sqrt(num_keypoints))
    keypoint_grid_width = int(np.ceil(num_keypoints/keypoint_grid_height))

    f, axs = plt.subplots(keypoint_grid_height,keypoint_grid_width)


    keypoint_plots = []
    for i in range(len(cfg.KEYPOINT_NAMES)):
        fig, (ax_hist, ax_box) = plt.subplots(2, sharex=True,
                    gridspec_kw= {"height_ratios": (1, 0.2)},
                    figsize=(6.92913,6.92913 / 1.618))
        keypoint_plots.append([fig, (ax_hist, ax_box)])


    gt_path = os.path.join(path, 'points_GroundTruth.csv')
    net_path = os.path.join(path, 'points_HybridNet.csv')

    pointsGT = np.genfromtxt(gt_path, delimiter=',')
    pointsGT = pointsGT.reshape(-1,int(pointsGT.shape[1]/3), 3)
    pointsNet = np.genfromtxt(net_path, delimiter=',')
    pointsNet = pointsNet.reshape(-1,int(pointsNet.shape[1]/3), 3)

    for keypoint in range(len(cfg.KEYPOINT_NAMES)):
        distances_l = {}
        distances = np.sqrt(np.sum(
                    (pointsNet[:,keypoint]-pointsGT[:,keypoint])**2, axis = 1))
        mask = np.sum(pointsGT[:,keypoint],axis = 1)
        distances = distances[mask != 0]
        if cutoff != -1:
            distances[distances>cutoff] = cutoff
        distances_l[cfg.KEYPOINT_NAMES[keypoint]] = (distances.reshape(-1))
        distances_pd = pd.DataFrame(distances_l)

        sns.histplot(data = distances_pd,
                    ax = axs[int(keypoint/keypoint_grid_width),
                    int(keypoint%keypoint_grid_width)],
                    element="step",alpha=0.1)

        ax_hist = keypoint_plots[keypoint][1][0]
        ax_box = keypoint_plots[keypoint][1][1]
        sns.boxplot(data = distances_pd, fliersize = 0, ax=ax_box, orient="h")
        sns.histplot(data = distances_pd, ax = ax_hist, element="step",
                    alpha=0.1)
        keypoint_plots[keypoint][0].savefig(os.path.join(path,
                    "keypoint_histograms",
                    f"{cfg.KEYPOINT_NAMES[keypoint]}.png"))
        plt.close(keypoint_plots[keypoint][0])

    if interactive:
        plt.show()
