"""
dataset3D.py
============
HybridNet 3D dataset loader.
"""

import os,sys,inspect
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
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
#from lib.dataset.utils import SetupVisualizer

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
    def __init__(self, cfg, set='train', **kwargs):
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
                    calibPaths[cam] = self.dataset['calibrations'][calibParams][cam]
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
            all_camera_names = [cam for cam in list(self.dataset['calibrations'].values())[0]]
            camera_names = [cam for cam in list(self.reproTools.values())[0].cameras]
            self.use_idxs = [i for i,cam in enumerate(all_camera_names) if cam in camera_names]



        for set in self.dataset['framesets']:
            keypoints3D_cam = np.zeros((cfg.KEYPOINTDETECT.NUM_JOINTS, 3))
            keypoints3D_bb = []
            file_name = self.imgs[self.dataset['framesets'][set]['frames'][0]]['file_name']
            info_split = file_name.split("/")
            key = info_split[0]
            for i in range(1,len(info_split)-2):
                key = key + "/" + info_split[i]
            key = key + "/" + info_split[-1].split(".")[0]
            frameset_ids = self.dataset['framesets'][key]['frames']
            if self.cameras_to_use != None:
                frameset_ids = [frameset_ids[i] for i in self.use_idxs]
            # if (len(info_split) == 4):
            #     frameset_ids = self.dataset['framesets'][info_split[0] + "/"
            #                 + info_split[1] + "/" + info_split[3].split(".")[0]]['frames']
            # elif (len(info_split) == 5):
            #     frameset_ids = self.dataset['framesets'][info_split[0] + "/"
            #                 + info_split[1] + "/" + info_split[2] + "/" +
            #                 info_split[4].split(".")[0]]['frames']
            keypoints_l = []
            for i,img_id in enumerate(frameset_ids):
                _, keypoints = self._load_annotations(img_id, is_id = True)
                keypoints = keypoints.reshape([-1, 3])
                keypoints_l.append(keypoints)
            for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS):
                points2D = np.zeros((self.num_cameras,2))
                camsToUse = []
                for cam in range(self.num_cameras):
                    if keypoints_l[cam][i][0] != 0 or keypoints_l[cam][i][1] != 0:
                        points2D[cam] = keypoints_l[cam][i][:2]
                        camsToUse.append(cam)
                keypoints3D_cam[i] = self.reproTools[self.dataset['framesets'][set]['datasetName']].reconstructPoint(
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
            if ((self.cfg.HYBRIDNET.ROI_CUBE_SIZE == None) or min_cube_size <= self.cfg.HYBRIDNET.ROI_CUBE_SIZE) and len(keypoints3D_bb) >1:
                self.image_ids.append(self.dataset['framesets'][set]['frames'][0])
                self.keypoints3D.append(keypoints3D_cam)
            # else:
            #     print ("Not adding the following frame to the set:")
            #     print (file_name)
            #     print (min_cube_size)

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
            if (keypoints3D[i][0] != 0 or keypoints3D[i][1] == 0 or keypoints3D[i][2] != 0):
                heatmap3D[i,xx,yy,zz] = 255.*np.exp(-0.5*(
                              np.power((keypoints3D_crop[i][0]-xx)/(exponent),2)
                            + np.power((keypoints3D_crop[i][1]-yy)/(exponent),2)
                            + np.power((keypoints3D_crop[i][2]-zz)/(exponent),2)))
        sample = [img_l, keypoints3D, centerHM, center3D, heatmap3D,
                    self.reproTools[datasetName].cameraMatrices,
                    self.reproTools[datasetName].intrinsicMatrices,
                    self.reproTools[datasetName].distortionCoefficients]

        return self.transform(sample)


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
        keypoints3D = np.array(self.keypoints3D)
        x_range = [np.min(keypoints3D[:,:,0]), np.max(keypoints3D[:,:,0])]
        y_range = [np.min(keypoints3D[:,:,1]), np.max(keypoints3D[:,:,1])]
        z_range = [np.min(keypoints3D[:,:,2]), np.max(keypoints3D[:,:,2])]
        tracking_area = np.array([x_range, y_range, z_range])
        x_cube_size_min = np.percentile(np.max(keypoints3D[:,:,0], axis = 1)
                        - np.min(keypoints3D[:,:,0],axis = 1),98)
        y_cube_size_min = np.percentile(np.max(keypoints3D[:,:,1], axis = 1)
                        - np.min(keypoints3D[:,:,1],axis = 1),98)
        z_cube_size_min = np.percentile(np.max(keypoints3D[:,:,2], axis = 1)
                        - np.min(keypoints3D[:,:,2],axis = 1),98)
        min_cube_size = np.max([x_cube_size_min,
                                y_cube_size_min,
                                z_cube_size_min])
        final_bbox_suggestion = int(np.ceil((min_cube_size*1.2)/8)*8)
        resolution_suggestion = int(np.floor((final_bbox_suggestion/100.)))

        suggested_parameters = {
            'bbox': final_bbox_suggestion,
            'resolution': resolution_suggestion,
        }

        # if show_visualization:
        #     figure = plt.figure()
        #     axes = figure.gca(projection='3d')
        #     visualizer = SetupVisualizer(
        #                 self.dataset['calibration']['primary_camera'],
        #                 self.root_dir,
        #                 self.dataset['calibration']['intrinsics'],
        #                 self.dataset['calibration']['extrinsics'])
        #     visualizer.plot_cameras(axes)
        #     visualizer.plot_tracking_area(tracking_area, axes)
        #     visualizer.plot_datapoints(self.keypoints3D[0], axes)
        #     plt.show()

        return suggested_parameters


    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        margin = 20 #Margin between pictures in pixels
        w = 4 # Width of the matrix (nb of images)
        h = 3 # Height of the matrix (nb of images)
        n = w*h
        out_size = (1000,1000)

        x,y,z=zip(*sample[1])
        center3D = sample[3]
        #grid_size = lookup_subset.shape[0]
        xx,yy,zz = np.meshgrid(np.arange(52),
                               np.arange(52),
                               np.arange(52), indexing='ij')
        #heatmap3D = self.get_heatmap_value(lookup_subset,
        #            np.transpose(np.array(sample[3]), axes = [1,0,2,3]),
        #            xx,yy,zz)
        heatmap3D = sample[4]

        figure = plt.figure()
        axes = figure.gca(projection='3d')


        colors = [(0,0,1,max(0,3./2*c-0.5)) for c in np.linspace(0,1,100)]
        cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap',
                    colors, N=100)

        points3D_hm = []
        for heatmap in heatmap3D:
            mean = [0,0,0]
            norm = 0
            for x in range(heatmap3D.shape[1]):
                for y in range(heatmap3D.shape[2]):
                    for z in range(heatmap3D.shape[3]):
                        if heatmap[x,y,z] > 128:
                            norm += heatmap[x,y,z]
                            mean[0] += ((x*self.cfg.HYBRIDNET.GRID_SPACING
                                        + center3D[0])-100)*heatmap[x,y,z]
                            mean[1] += ((y*self.cfg.HYBRIDNET.GRID_SPACING
                                        + center3D[1])-100)*heatmap[x,y,z]
                            mean[2] += ((z*self.cfg.HYBRIDNET.GRID_SPACING
                                        + center3D[2])-100)*heatmap[x,y,z]
            points3D_hm.append(np.array(mean)/norm)


        colors = ['r', 'r','r','r','b','b','b','b','g','g','g','g',
             'orange', 'orange','orange','orange', 'y','y','y','y',
             'purple', 'purple','purple']
        #colors = ['r', 'r', 'b','b','b', 'b', 'gray', 'y','y', 'purple', 'purple', 'orange']
        colors = ['r', 'r','r', 'gray', 'b', 'b', 'b', 'b', 'g', 'g','g','g', 'gray', 'y', 'y','y', 'purple', 'purple','purple', 'orange', 'orange', 'orange']
        #colors = ['r', 'r','r', 'r', 'gray', 'b', 'b', 'b', 'b', 'g', 'g','g','g', 'gray', 'y', 'y','y','y', 'purple', 'purple','purple','purple', 'orange', 'orange', 'orange']
        line_idxs = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [8,9], [9,10],
                     [10,11], [12,13], [13,14], [14,15], [16,17], [17,18],
                     [18,19], [3,7], [7,11], [11,15], [3,21], [7,21],[11,22],
                     [15,22],[21,22], [18,15], [19,22]]
        #line_idxs = [[0,1], [1,2], [0,2], [2,3], [0,3], [1,3]]
        #line_idxs = [[4,5], [5,6], [6,7], [7,8], [4,9], [9,10], [10,11], [11,12],
        #             [4,13],[13,14], [14,15], [15,16], [16,17], [13,18], [18,19]]


        keypoins3D = sample[1]
        #print (np.linalg.norm(keypoins3D[2]-keypoins3D[3]))
        for i, point in enumerate(keypoins3D):
            if i != 99:
                if (point[0] != 0):
                    axes.scatter(point[0], point[1], point[2], color = colors[i])
        #print ("Lengths:")
        for line in line_idxs:
            #print (np.linalg.norm(keypoins3D[line[0]]-keypoins3D[line[1]]))
            if (keypoins3D[line[0]][0] != 0 and keypoins3D[line[1]][0] != 0):
                axes.plot([keypoins3D[line[0]][0], keypoins3D[line[1]][0]],
                          [keypoins3D[line[0]][1], keypoins3D[line[1]][1]],
                          [keypoins3D[line[0]][2], keypoins3D[line[1]][2]],
                          c = 'gray')

        cube = np.array(list(itertools.product(*zip([-200,-200,-200],[200,200,200]))))
        cube = cube + center3D
        for corner in cube:
            axes.scatter(corner[0],corner[1],corner[2], c = 'magenta', s = 20)

        vertices = [[0,1], [0,2], [2,3], [1,3], [4,5], [4,6], [6,7], [5,7],
                    [0,4], [1,5], [2,6], [3,7]]
        for vertex in vertices:
            axes.plot([cube[vertex[0]][0], cube[vertex[1]][0]],
                      [cube[vertex[0]][1], cube[vertex[1]][1]],
                      [cube[vertex[0]][2], cube[vertex[1]][2]],c = 'magenta')

        plt.show()



class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        return [(sample[0].astype(np.float32) - self.mean) / self.std,
                 sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7]]



if __name__ == "__main__":
    from jarvis.config.project_manager import ProjectManager

    project = ProjectManager()
    project.load('Pancake')

    cfg = project.get_cfg()
    idx = 0
    training_set = Dataset3D(cfg = cfg, set='train')#, cameras_to_use = ['Camera_T', 'Camera_B', 'Camera_LBB'])
    #training_set.get_dataset_config(True)
    #print (len(training_set))
    for i in range(100):
        training_set.visualize_sample(i)
    # frameNames = []
    # for i in range(len(training_set.image_ids)):
    #     image_info = training_set.coco.loadImgs(training_set.image_ids[i])[0]
    #     frameNames.append(image_info['file_name'])
    #
    # import csv
    # with open("framesNames.csv", 'w', newline='') as myfile:
    #  wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
    #  for name in frameNames:
    #      wr.writerow([name])
    #
    # #training_set.get_dataset_config(show_visualization = True)
    # for i in range(0,121):
    #     training_set.visualize_sample(i)
    # val_frame_numbers = []
    # print (len(training_set.image_ids))
