"""
dataset3D.py
============
HybridNet 3D dataset loader.
"""

import os,sys,inspect
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from jarvis.dataset.datasetBase import BaseDataset
from jarvis.dataset.utils import ReprojectionTool

class Dataset3D(BaseDataset):
    """
    Dataset Class to load 3D datasets in the HybridNet dataset format, inherits
    from BaseDataset class.

    :param cfg: handle of the global configuration
    :param set: specifies wether to load training ('train') or validation
                ('val') split. Augmentation will only be applied to training
                split.
    :type set: string
    """
    def __init__(self, cfg, set='train', analysisMode = False, **kwargs):
        self.analysisMode = analysisMode
        self.cameras_to_use = None
        if 'cameras_to_use' in kwargs:
            self.cameras_to_use = kwargs['cameras_to_use']
        dataset_name = cfg.DATASET.DATASET_3D
        super().__init__(cfg, dataset_name, set, **kwargs)

        img = self._load_image(0)
        width, height = img.shape[1], img.shape[0]
        cfg.DATASET.IMAGE_SIZE = [width,height]

        self.reproTools = {}
        for calibParams in self.dataset['calibrations']:
            calibPaths = {}
            for cam in self.dataset['calibrations'][calibParams]:
                if self.cameras_to_use == None or cam in self.cameras_to_use:
                    calibPaths[cam] = self.dataset['calibrations'] \
                                [calibParams][cam]
            self.reproTools[calibParams] = ReprojectionTool(
                        os.path.join(cfg.PARENT_DIR, self.root_dir), calibPaths)
            self.num_cameras = self.reproTools[calibParams].num_cameras
            self.reproTools[calibParams].resolution = [width,height]

        cfg.HYBRIDNET.NUM_CAMERAS = self.num_cameras

        self.image_ids_all = self.image_ids
        self.image_ids = []
        self.keypoints3D = []
        cfg.KEYPOINTDETECT.NUM_JOINTS = self.num_keypoints[0]

        if self.cameras_to_use != None:
            all_camera_names = [cam for cam in list(
                        self.dataset['calibrations'].values())[0]]
            camera_names = [cam for cam in list(
                        self.reproTools.values())[0].cameras]
            self.use_idxs = [i for i,cam in enumerate(all_camera_names)
                        if cam in camera_names]



        for set in self.dataset['framesets']:
            keypoints3D_cam = np.zeros((cfg.KEYPOINTDETECT.NUM_JOINTS, 3))
            keypoints3D_bb = []
            file_name = self.imgs[self.dataset['framesets'][set]['frames'][0]] \
                        ['file_name']
            info_split = file_name.split("/")
            key = info_split[0]
            for i in range(1,len(info_split)-2):
                key = key + "/" + info_split[i]
            key = key + "/" + info_split[-1].split(".")[0]
            frameset_ids = self.dataset['framesets'][key]['frames']
            if self.cameras_to_use != None:
                frameset_ids = [frameset_ids[i] for i in self.use_idxs]
            keypoints_l = []
            for i,img_id in enumerate(frameset_ids):
                _, keypoints = self._load_annotations(img_id, is_id = True)
                keypoints = keypoints.reshape([-1, 3])
                keypoints_l.append(keypoints)
            for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS):
                points2D = np.zeros((self.num_cameras,2))
                camsToUse = []
                for cam in range(self.num_cameras):
                    if (keypoints_l[cam][i][0] != 0
                                or keypoints_l[cam][i][1] != 0):
                        points2D[cam] = keypoints_l[cam][i][:2]
                        camsToUse.append(cam)
                keypoints3D_cam[i] = self.reproTools[self.dataset['framesets'] \
                            [set]['datasetName']].reconstructPoint(
                            points2D.transpose(), camsToUse)
                if len(camsToUse) > 1:
                    keypoints3D_bb.append(keypoints3D_cam[i])
            if len(keypoints3D_bb) == 0:
                keypoints3D_bb.append([0,0,0])
            keypoints3D_bb = np.array(keypoints3D_bb)
            x_range = [np.min(keypoints3D_bb[:,0]),
                       np.max(keypoints3D_bb[:,0])]
            y_range = [np.min(keypoints3D_bb[:,1]),
                       np.max(keypoints3D_bb[:,1])]
            z_range = [np.min(keypoints3D_bb[:,2]),
                       np.max(keypoints3D_bb[:,2])]
            tracking_area = np.array([x_range, y_range, z_range])
            x_cube_size_min = np.max(np.max(keypoints3D_bb[:,0], axis = 0)
                            - np.min(keypoints3D_bb[:,0],axis = 0))
            y_cube_size_min = np.max(np.max(keypoints3D_bb[:,1], axis = 0)
                            - np.min(keypoints3D_bb[:,1],axis = 0))
            z_cube_size_min = np.max(np.max(keypoints3D_bb[:,2], axis = 0)
                            - np.min(keypoints3D_bb[:,2],axis = 0))
            min_cube_size = np.max([x_cube_size_min,
                                    y_cube_size_min,
                                    z_cube_size_min])
            if (((self.cfg.HYBRIDNET.ROI_CUBE_SIZE == None)
                        or min_cube_size <= self.cfg.HYBRIDNET.ROI_CUBE_SIZE)
                        and len(keypoints3D_bb) > 1):
                self.image_ids.append(
                            self.dataset['framesets'][set]['frames'][0])
                self.keypoints3D.append(keypoints3D_cam)
            # else:
            #      print (min_cube_size)

        self.transform = transforms.Compose(
                    [Normalizer(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)])
        self._build_augpipe()


    def _build_augpipe(self):
        augmentors = []
        cfg = self.cfg.AUGMENTATION
        if cfg.COLOR_MANIPULATION.ENABLED:
            cman = cfg.COLOR_MANIPULATION
            augmentors.append(
                iaa.Sometimes(cman.GAUSSIAN_BLUR.PROBABILITY,
                iaa.GaussianBlur(sigma=cman.GAUSSIAN_BLUR.SIGMA)))
            augmentors.append(
                iaa.AdditiveGaussianNoise(loc = 0,
                scale = cman.GAUSSIAN_NOISE.SCALE,
                per_channel = cman.GAUSSIAN_NOISE.PER_CHANNEL_PROBABILITY))
            augmentors.append(
                iaa.Sometimes(cman.LINEAR_CONTRAST.PROBABILITY,
                iaa.LinearContrast(cman.LINEAR_CONTRAST.SCALE)))
            augmentors.append(
                iaa.Sometimes(cman.MULTIPLY.PROBABILITY,
                iaa.Multiply(cman.MULTIPLY.SCALE)))
            augmentors.append(
                iaa.Sometimes(cman.PER_CHANNEL_MULTIPLY.PROBABILITY,
                iaa.Multiply(cman.PER_CHANNEL_MULTIPLY.SCALE,
                per_channel=cman.PER_CHANNEL_MULTIPLY.PER_CHANNEL_PROBABILITY)))

        self.augpipe = iaa.Sequential(augmentors, random_order = False)

    def __getitem__(self, idx):
        grid_spacing = self.cfg.HYBRIDNET.GRID_SPACING
        grid_size = self.cfg.HYBRIDNET.ROI_CUBE_SIZE
        file_name = self.imgs[self.image_ids[idx]]['file_name']

        info_split = file_name.split("/")
        key = info_split[0]
        for i in range(1,len(info_split)-2):
            key = key + "/" + info_split[i]
        key = key + "/" + info_split[-1].split(".")[0]
        frameset_ids = self.dataset['framesets'][key]['frames']
        if self.cameras_to_use != None:
            frameset_ids = [frameset_ids[i] for i in self.use_idxs]
        datasetName = self.dataset['framesets'][key]['datasetName']
        img_l = np.zeros((self.num_cameras,
                          self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                          self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,3))
        if self.analysisMode:
            img = self._load_image(0)
            img_l = np.zeros((self.num_cameras,img.shape[0],img.shape[1],3))

        centerHM = np.full((self.num_cameras, 2), 128, dtype = int)
        bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)

        for frame_idx,img_id in enumerate(frameset_ids):
            img = self._load_image(img_id, is_id = True)
            img_shape = (img.shape[:-1])
            bboxs, _ = self._load_annotations(img_id, is_id = True)
            center_y = int((bboxs[0,1]+int(bboxs[0,3]))/2)
            center_x = int((bboxs[0,0]+int(bboxs[0,2]))/2)
            if self.set_name == 'train':
                translation_factors = np.random.uniform(-1.,1.,2)
                center_x += int(translation_factors[0]*bbox_hw*0.3)
                center_y += int(translation_factors[1]*bbox_hw*0.3)
            center_y = min(max(bbox_hw, center_y), img_shape[0]-bbox_hw)
            center_x = min(max(bbox_hw, center_x), img_shape[1]-bbox_hw)
            centerHM[frame_idx] = np.array([center_x, center_y])
            if not self.analysisMode:
                img = img[center_y-bbox_hw:center_y+bbox_hw,
                          center_x-bbox_hw:center_x+bbox_hw, :]
            if self.set_name == 'train':
                img = self.augpipe(image=img)
            img_l[frame_idx] = img

        keypoints3D = self.keypoints3D[idx]
        x,y,z=zip(*keypoints3D)
        x = [xx for xx in x if xx != 0]
        y = [yy for yy in y if yy != 0]
        z = [zz for zz in z if zz != 0]
        center3D = np.array([
                    int((max(x)+min(x))/float(grid_spacing)/2.)*grid_spacing,
                    int((max(y)+min(y))/float(grid_spacing)/2.)*grid_spacing,
                    int((max(z)+min(z))/float(grid_spacing)/2.)*grid_spacing])

        if self.set_name == 'train':
            translation_margins = np.array([grid_size-(max(x)-min(x)),
                                            grid_size-(max(y)-min(y)),
                                            grid_size-(max(z)-min(z))])
            translation_factors = np.random.uniform(-0.4,0.4,3)
            center3D += (np.array((translation_margins*translation_factors)
                        / float(grid_spacing)/2., dtype = int) * grid_spacing)

        keypoints3D_crop = ((keypoints3D+float(grid_size/2.)-center3D)
                            / float(grid_spacing)/2.)

        heatmap_size = int(grid_size/grid_spacing/2.)
        heatmap3D = np.zeros((self.cfg.KEYPOINTDETECT.NUM_JOINTS,
                              heatmap_size,heatmap_size,heatmap_size))

        xx,yy,zz = np.meshgrid(np.arange(heatmap_size),
                               np.arange(heatmap_size),
                               np.arange(heatmap_size))

        exponent = 1.7/(float(2)/2)
        for i in range(self.cfg.KEYPOINTDETECT.NUM_JOINTS):
            if (keypoints3D[i][0] != 0 or keypoints3D[i][1] == 0
                        or keypoints3D[i][2] != 0):
                heatmap3D[i,xx,yy,zz] = 255.*np.exp(-0.5*(
                          np.power((keypoints3D_crop[i][0]-xx)/(exponent),2)
                        + np.power((keypoints3D_crop[i][1]-yy)/(exponent),2)
                        + np.power((keypoints3D_crop[i][2]-zz)/(exponent),2)))
        sample = [img_l, keypoints3D, centerHM, center3D, heatmap3D,
                    self.reproTools[datasetName].cameraMatrices,
                    self.reproTools[datasetName].intrinsicMatrices,
                    self.reproTools[datasetName].distortionCoefficients,
                                datasetName]

        if not self.analysisMode:
            sample = self.transform(sample)
        else:
            sample = sample + [file_name]
        return sample


    def __len__(self):
        return len(self.image_ids)

    def get_dataset_config(self, show_visualization = False):
        """
        Get the recommended configuration for the 3D Dataset. Recommendations
        are computed by analyzing the trainingset, if it is not representative
        of the data you plan to analyze, parameters might have to be adjusted.

        :param show_visualization: Show a 3D plot visualizing the camera
                                   configuration and tracking volume
        :type show_visualization: string
        """
        #keypoints3D = np.array(self.keypoints3D)
        tracking_areas = []
        for i,keypoints in enumerate(self.keypoints3D):
            keypoints3D_filtered = []
            for keypoint in keypoints:
                if keypoint[0] != 0 or keypoint[1] != 0 or keypoint[2] != 0:
                    keypoints3D_filtered.append(keypoint)
            keypoints3D = np.array(keypoints3D_filtered)
            x_range = [np.min(keypoints3D[:,0]), np.max(keypoints3D[:,0])]
            y_range = [np.min(keypoints3D[:,1]), np.max(keypoints3D[:,1])]
            z_range = [np.min(keypoints3D[:,2]), np.max(keypoints3D[:,2])]
            tracking_area = np.array([x_range, y_range, z_range])
            tracking_areas.append(tracking_area)
        tracking_areas = np.array(tracking_areas)
        x_cube_size_min = np.percentile(tracking_areas[:,0,1]
                    - tracking_areas[:,0,0],95)
        y_cube_size_min = np.percentile(tracking_areas[:,1,1]
                    - tracking_areas[:,1,0],95)
        z_cube_size_min = np.percentile(tracking_areas[:,2,1]
                    - tracking_areas[:,2,0],95)
        min_cube_size = np.max([x_cube_size_min,
                                y_cube_size_min,
                                z_cube_size_min])
        rough_bbox_suggestion = min_cube_size*1.25
        resolution_suggestion = max(1,int(np.round_(
                    rough_bbox_suggestion / 85.)))
        final_bbox_suggestion = int(np.ceil((min_cube_size * 1.25)
                    / (resolution_suggestion * 4)) * resolution_suggestion * 4)

        suggested_parameters = {
            'bbox': final_bbox_suggestion,
            'resolution': resolution_suggestion,
        }

        return suggested_parameters



class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        return [(sample[0].astype(np.float32) - self.mean) / self.std,
                 sample[1], sample[2], sample[3], sample[4], sample[5],
                 sample[6], sample[7], sample[8]]



if __name__ == "__main__":
    from jarvis.config.project_manager import ProjectManager

    project = ProjectManager()
    project.load('ExR_Feeder')

    cfg = project.get_cfg()
    idx = 0
    training_set = Dataset3D(cfg = cfg, set='val')#, cameras_to_use = ['Camera_T', 'Camera_B', 'Camera_LBB'])
    #training_set.get_dataset_config(True)
    #print (len(training_set))
    # print (len(training_set.image_ids))
