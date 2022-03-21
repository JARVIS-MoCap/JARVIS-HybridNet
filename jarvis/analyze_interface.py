import os
import time
import cv2
import numpy as np
from numpy import savetxt
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st



from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.efficienttrack.efficienttrack import EfficientTrack
import jarvis.efficienttrack.darkpose as darkpose
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.prediction.predict3D import load_reprojection_tools



def plot_error_histogram(path, additional_data = {}, cutoff = -1):
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


    plt.xlabel('Deviation from hand annotations [mm]')
    if cutoff != -1:
        if cutoff < 15:
            step = 2
        else:
            step = 5
        plt.xlim(0,cutoff+0.1)
        x_labels = [str(i) for i in range(0,cutoff, step)] + [f'>{cutoff}']
        plt.xticks(list(step * np.arange(len(x_labels)-1))+[cutoff])
        ax_box.set_xticklabels(x_labels)
    plt.show()


def plot_error_per_keypoint(path):
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
    joint_labels = [str(i) for i in range(number_joints)]

    cmap = plt.cm.get_cmap('jet')
    for i in range(0,len(joints)):
        plt.bar(joints[i], joints_means[i],
                    width = barWidth, color = cmap(i*(1/number_joints)))

    plt.xticks([r-1 + barWidth for r in range(len(joints))],
                joint_labels, rotation=90)
    plt.show()



def analyze_validation_data(project_name, weights_center = 'latest',
            weights_hybridnet = 'latest', cameras_to_use = None, progress_bar = None):
    project = ProjectManager()
    project.load(project_name)
    cfg = project.get_cfg()

    output_dir = os.path.join(project.parent_dir,
                project.cfg.PROJECTS_ROOT_PATH, project_name,
                'analysis', f'Validation_Predictions_'
                f'{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(output_dir)

    dataset = Dataset3D(cfg = cfg, set='val', analysisMode = True,
                cameras_to_use = cameras_to_use)
    hybridNet = HybridNet('inference', project.cfg, weights_hybridnet)
    centerDetect = EfficientTrack('CenterDetectInference', project.cfg,
                weights_center)
    reproTools = load_reprojection_tools(cfg, cameras_to_use = cameras_to_use)

    img_downsampled_shape = centerDetect.cfg.IMAGE_SIZE
    def process_images(img):
        img = cv2.resize(img, (img_downsampled_shape,img_downsampled_shape))
        return img

    pointsNet = []
    pointsGT = []

    for item in tqdm(range(len(dataset.image_ids))):
        if progress_bar != None:
            progress_bar.progress(float(item+1)/len(dataset.image_ids))
        sample = dataset.__getitem__(item)
        keypoints3D = sample[1]
        imgs_orig = sample[0]
        img_size = imgs_orig[0].shape
        dataset_name = sample[-1]
        reproTool = reproTools[dataset_name]
        num_cameras = imgs_orig.shape[0]

        centerHMs = []
        camsToUse = []
        downsampling_scale = np.array([
                    float(imgs_orig[0].shape[1]/img_downsampled_shape),
                    float(imgs_orig[0].shape[0]/img_downsampled_shape)])
        imgs = Parallel(n_jobs=-1, require='sharedmem')(
                    delayed(process_images, )(img) for img in imgs_orig)

        imgs = torch.from_numpy(np.array(imgs).transpose(0,3,1,2)).cuda().float()
        outputs = centerDetect.model(imgs)
        preds, maxvals = darkpose.get_final_preds(
                    outputs[1].clamp(0,255).detach().cpu().numpy(), None)
        camsToUse = []

        for i,val in enumerate(maxvals[:]):
            if val > 100:
                camsToUse.append(i)
        if len(camsToUse) >= 2:
            center3D = torch.from_numpy(reproTool.reconstructPoint((
                        preds.reshape(num_cameras,2)*(downsampling_scale*2)).transpose(),
                        camsToUse))
            reproPoints = reproTool.reprojectPoint(center3D)

            errors = []
            errors_valid = []
            for i in range(num_cameras):
                if maxvals[i] > 100:
                    errors.append(np.linalg.norm(preds.reshape(
                                num_cameras,2)[i]*downsampling_scale*2-reproPoints[i]))
                    errors_valid.append(np.linalg.norm(preds.reshape(
                                num_cameras,2)[i]*downsampling_scale*2-reproPoints[i]))
                else:
                    errors.append(0)
            medianError = np.median(np.array(errors_valid))
            camsToUse = []
            for i,val in enumerate(maxvals[:]):
                if val > 100 and errors[i] < 2*medianError:
                    camsToUse.append(i)
            center3D = torch.from_numpy(reproTool.reconstructPoint(
                        (preds.reshape(num_cameras,2)*downsampling_scale*2).transpose(),
                        camsToUse))
            reproPoints = reproTool.reprojectPoint(center3D)
            imgs = []
            bbox_hw = int(hybridNet.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
            for idx,reproPoint in enumerate(reproPoints):
                reproPoint = reproPoint.astype(int)
                reproPoints[idx][0] = min(max(reproPoint[0], bbox_hw),
                            img_size[1]-1-bbox_hw)
                reproPoints[idx][1] = min(max(reproPoint[1], bbox_hw),
                            img_size[0]-1-bbox_hw)
                reproPoint[0] = reproPoints[idx][0]
                reproPoint[1] = reproPoints[idx][1]
                img = imgs_orig[idx][reproPoint[1]-bbox_hw:reproPoint[1]+bbox_hw,
                            reproPoint[0]-bbox_hw:reproPoint[0]+bbox_hw, :]
                imgs.append(img)


            imgs = torch.from_numpy(np.array(imgs))
            imgs = imgs.permute(0,3,1,2).view(1,num_cameras,3,
                        hybridNet.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                        hybridNet.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda().float()
            centerHMs = np.array(reproPoints).astype(int)

        if len(camsToUse) >= 2:
            center3D = center3D.int().cuda()
            centerHMs = torch.from_numpy(centerHMs).cuda()
            heatmap3D, heatmaps_padded, points3D_net = hybridNet.model(imgs,
                        torch.tensor(img_size).cuda(),
                        torch.unsqueeze(centerHMs,0),
                        torch.unsqueeze(center3D, 0),
                        torch.unsqueeze(reproTool.cameraMatrices.cuda(),0),
                        torch.unsqueeze(reproTool.intrinsicMatrices.cuda(),0),
                        torch.unsqueeze(reproTool.distortionCoefficients.cuda(),0))
            points3D_net = points3D_net[0].cpu().detach().numpy()
            pointsNet.append(points3D_net)
            pointsGT.append(keypoints3D)

    print (f'{CLIColors.OKGREEN}Successfully analysed all validation '
                f'frames!{CLIColors.ENDC}')
    if len(pointsNet) != len(dataset.image_ids):
        print (f'{CLIColors.WARNING}Network could not detect instance in '
                    f'{len(dataset.image_ids) - len(pointsNet)} frameSets. '
                    f'Those were not included in the output '
                    f'files!{CLIColors.ENDC}')

    savetxt(os.path.join(output_dir, 'points_HybridNet.csv'),
                np.array(pointsNet).reshape(
                (-1, hybridNet.cfg.KEYPOINTDETECT.NUM_JOINTS*3)), delimiter=',')
    savetxt(os.path.join(output_dir, 'points_GroundTruth.csv'),
                np.array(pointsGT).reshape(
                (-1, hybridNet.cfg.KEYPOINTDETECT.NUM_JOINTS*3)), delimiter=',')
