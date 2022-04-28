import os
import time
import numpy as np
from numpy import savetxt
from tqdm import tqdm

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
from jarvis.utils.reprojection import load_reprojection_tools
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.prediction.jarvis3D import JarvisPredictor3D
from torch.utils.data import DataLoader


def analyze_validation_data(project_name, weights_center = 'latest',
            weights_hybridnet = 'latest', cameras_to_use = None,
            progress_bar = None):
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

    jarvisPredictor = JarvisPredictor3D(project.cfg, weights_center,
                weights_hybridnet)

    reproTools = load_reprojection_tools(cfg, cameras_to_use = cameras_to_use)

    pointsNet = []
    pointsGT = []
    filenames = []
    data_generator = DataLoader(
                dataset,
                batch_size = 1,
                shuffle = False,
                num_workers =  cfg.DATALOADER_NUM_WORKERS,
                pin_memory = True)


    for item, sample in enumerate(tqdm(data_generator)):
        if progress_bar != None:
            progress_bar.progress(float(item+1)/len(dataset.image_ids))

        keypoints3D = sample[1][0].numpy()
        imgs_orig = sample[0][0]
        img_size = imgs_orig[0].shape
        num_cameras = imgs_orig.shape[0]
        dataset_name = sample[-2][0]
        reproTool = reproTools[dataset_name]
        file_name = sample[-1][0]

        imgs = imgs_orig.cuda().float().permute(0,3,1,2)

        points3D_net = jarvisPredictor(imgs,
                    reproTool.cameraMatrices.cuda(),
                    reproTool.intrinsicMatrices.cuda(),
                    reproTool.distortionCoefficients.cuda())

        if points3D_net != None:
            points3D_net = points3D_net[0].cpu().detach().numpy()
            pointsNet.append(points3D_net)
            pointsGT.append(keypoints3D)
            filenames.append(file_name)

    print (f'{CLIColors.OKGREEN}Successfully analysed all validation '
                f'frames!{CLIColors.ENDC}')
    if len(pointsNet) != len(dataset.image_ids):
        print (f'{CLIColors.WARNING}Network could not detect instance in '
                    f'{len(dataset.image_ids) - len(pointsNet)} frameSets. '
                    f'Those were not included in the output '
                    f'files!{CLIColors.ENDC}')


    savetxt(os.path.join(output_dir, 'frame_names.csv'),
                np.array(filenames), delimiter=',', fmt='%s')

    savetxt(os.path.join(output_dir, 'points_HybridNet.csv'),
                np.array(pointsNet).reshape(
                (-1, project.cfg.KEYPOINTDETECT.NUM_JOINTS*3)), delimiter=',')
    savetxt(os.path.join(output_dir, 'points_GroundTruth.csv'),
                np.array(pointsGT).reshape(
                (-1, project.cfg.KEYPOINTDETECT.NUM_JOINTS*3)), delimiter=',')
