import os
import time
import numpy as np
from numpy import savetxt
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
from jarvis.utils.reprojection import load_reprojection_tools
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.prediction.jarvis3D import JarvisPredictor3D


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

    jarvisPredictor = JarvisPredictor3D(project.cfg, weights_center, weights_hybridnet)

    reproTools = load_reprojection_tools(cfg, cameras_to_use = cameras_to_use)

    pointsNet = []
    pointsGT = []

    for item in tqdm(range(len(dataset.image_ids))):
        if progress_bar != None:
            progress_bar.progress(float(item+1)/len(dataset.image_ids))

        file_name = dataset.imgs[dataset.image_ids[item]]['file_name']

        sample = dataset.__getitem__(item)
        keypoints3D = sample[1]
        imgs_orig = sample[0]
        img_size = imgs_orig[0].shape
        dataset_name = sample[-1]
        reproTool = reproTools[dataset_name]
        num_cameras = imgs_orig.shape[0]

        imgs = torch.from_numpy(imgs_orig).cuda().float().permute(0,3,1,2)

        points3D_net = jarvisPredictor(imgs, reproTool.cameraMatrices.cuda(), reproTool.intrinsicMatrices.cuda(), reproTool.distortionCoefficients.cuda())

        if points3D_net != None:
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
                (-1, project.cfg.KEYPOINTDETECT.NUM_JOINTS*3)), delimiter=',')
    savetxt(os.path.join(output_dir, 'points_GroundTruth.csv'),
                np.array(pointsGT).reshape(
                (-1, project.cfg.KEYPOINTDETECT.NUM_JOINTS*3)), delimiter=',')
