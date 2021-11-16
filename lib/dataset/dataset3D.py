"""
dataset3D.py
============
Vortex 3D dataset loader.
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

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from lib.dataset.datasetBase import VortexBaseDataset
from lib.vortex.utils import ReprojectionTool
from lib.dataset.utils import SetupVisualizer

class VortexDataset3D(VortexBaseDataset):
    """
    Dataset Class to load 3D datasets in the VoRTEx dataset format, inherits from
    VortexBaseDataset class. See HERE for more details.

    :param cfg: handle of the global configuration
    :param set: specifies wether to load training ('train') or validation ('val') split.
                Augmentation will only be applied to training split.
    :type set: string
    """
    def __init__(self, cfg, set='train'):
        dataset_name = cfg.DATASET.DATASET_3D
        super().__init__(cfg, dataset_name, set)
        self.augpipe = iaa.Sequential([
            iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.7, 1))),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02), per_channel=0.6)),
            iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.2))),
            iaa.Sometimes(0.3, iaa.Multiply((0.8, 1.2), per_channel=1.0))],
            random_order=False)

        self.reproTool = ReprojectionTool(self.coco.dataset['calibration']['primary_camera'], self.root_dir, self.coco.dataset['calibration']['intrinsics'], self.coco.dataset['calibration']['extrinsics'])
        self.num_cameras = self.reproTool.num_cameras
        self.image_ids_all = self.image_ids
        self.image_ids = []
        self.keypoints3D = []
        for set in self.coco.dataset['framesets']:
            #if len(self.coco.dataset['framesets'][set]) == self.num_cameras:
            keypoints3D_cam = np.zeros((cfg.EFFICIENTTRACK.NUM_JOINTS, 3))
            image_info = self.coco.loadImgs(self.coco.dataset['framesets'][set][0])[0]
            info_split = image_info['file_name'].split("/")
            frameset_ids = self.coco.dataset['framesets'][info_split[0] + "/" + info_split[1] + "/" + info_split[3].split(".")[0]]
            keypoints_l = []
            for i,img_id in enumerate(frameset_ids):
                _, keypoints = self._load_annotations(img_id, is_id = True)
                keypoints = keypoints.reshape([-1, 3])
                keypoints_l.append(keypoints)
            for i in range(cfg.EFFICIENTTRACK.NUM_JOINTS):
                points2D = np.zeros((self.num_cameras,2))
                camsToUse = []
                for cam in range(self.num_cameras):
                    if keypoints_l[cam][i][2] != 0:
                        points2D[cam] = keypoints_l[cam][i][:2]
                        camsToUse.append(cam)
                keypoints3D_cam[i] = self.reproTool.reconstructPoint(points2D.transpose(), camsToUse)
            x_range = [np.min(keypoints3D_cam[:,0]), np.max(keypoints3D_cam[:,0])]
            y_range = [np.min(keypoints3D_cam[:,1]), np.max(keypoints3D_cam[:,1])]
            z_range = [np.min(keypoints3D_cam[:,2]), np.max(keypoints3D_cam[:,2])]
            tracking_area = np.array([x_range, y_range, z_range])
            x_cube_size_min = np.max(np.max(keypoints3D_cam[:,0], axis = 0)-np.min(keypoints3D_cam[:,0],axis = 0))
            y_cube_size_min = np.max(np.max(keypoints3D_cam[:,1], axis = 0)-np.min(keypoints3D_cam[:,1],axis = 0))
            z_cube_size_min = np.max(np.max(keypoints3D_cam[:,2], axis = 0)-np.min(keypoints3D_cam[:,2],axis = 0))
            min_cube_size = np.max([x_cube_size_min,y_cube_size_min,z_cube_size_min])
            if (min_cube_size < 208):
                self.image_ids.append(self.coco.dataset['framesets'][set][0])
                self.keypoints3D.append(keypoints3D_cam)
            else:
                print (image_info['file_name'])
                print (min_cube_size)
                self.image_ids.append(self.coco.dataset['framesets'][set][0])
                print (len(self.image_ids))
                self.keypoints3D.append(keypoints3D_cam)
                print (len(camsToUse))

        self.transform = transforms.Compose([Normalizer(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)])
        #self.get_dataset_config(self.keypoints3D)


    def __getitem__(self, idx):
        grid_spacing = self.cfg.VORTEX.GRID_SPACING
        grid_size = self.cfg.VORTEX.ROI_CUBE_SIZE
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        info_split = image_info['file_name'].split("/")
        frameset_ids = self.coco.dataset['framesets'][info_split[0] + "/" + info_split[1] + "/" + info_split[3].split(".")[0]]
        img_l = np.zeros((self.num_cameras, self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE,self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE,3))
        centerHM = np.full((self.num_cameras, 2), 128, dtype = int)
        bbox_hw = int(self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE/2)
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
            img = img[center_y-bbox_hw:center_y+bbox_hw, center_x-bbox_hw:center_x+bbox_hw, :]
            if self.set_name == 'train':
                img = self.augpipe(image=img)
            img_l[frame_idx] = img

        keypoints3D = self.keypoints3D[idx]
        x,y,z=zip(*keypoints3D)
        center3D = np.array([int((max(x)+min(x))/float(grid_spacing)/2.)*grid_spacing,
                             int((max(y)+min(y))/float(grid_spacing)/2.)*grid_spacing,
                             int((max(z)+min(z))/float(grid_spacing)/2.)*grid_spacing])

        if self.set_name == 'train':
            translation_margins = np.array([grid_size-(max(x)-min(x)),
                                            grid_size-(max(y)-min(y)),
                                            grid_size-(max(z)-min(z))])
            translation_factors = np.random.uniform(-0.4,0.4,3)
            center3D += np.array((translation_margins*translation_factors)/float(grid_spacing)/2., dtype = int) *2

        keypoints3D_crop = (keypoints3D+float(grid_size/grid_spacing)-center3D)/float(grid_spacing)/2.

        heatmap_size = int(grid_size/grid_spacing/2.)
        heatmap3D = np.zeros((self.cfg.EFFICIENTTRACK.NUM_JOINTS, heatmap_size,heatmap_size,heatmap_size))

        #grid = np.array(np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), np.arange(heatmap_size))).transpose(1,2,3,0)

        #for i in range(self.cfg.EFFICIENTTRACK.NUM_JOINTS):
        #    dist = (keypoints3D_crop[i]-grid)
        #    heatmap3D[i] = 255.*np.exp(-0.5*(np.sum(dist*dist/(1.7**2), axis = 3)))

        xx,yy,zz = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), np.arange(heatmap_size))
        for i in range(self.cfg.EFFICIENTTRACK.NUM_JOINTS):
            heatmap3D[i,xx,yy,zz] = 255.*np.exp(-0.5*(np.power((keypoints3D_crop[i][0]-xx)/(1.7),2)+np.power((keypoints3D_crop[i][1]-yy)/(1.7),2)+np.power((keypoints3D_crop[i][2]-zz)/(1.7),2)))

        sample = [img_l, keypoints3D, centerHM, center3D, heatmap3D]
        return self.transform(sample)


    def __len__(self):
        return len(self.image_ids)

    def get_dataset_config(self, show_visualization = False):
        """
        Get the recommended configuration for the 3D Dataset. Recommendations are
        computed by analyzing the trainingset, if it is not representative of the data
        you plan to analyze, parameters like the overall tracking volume might have to be adjusted.

        :param show_visualization: Show a 3D plot visualizing the camera configuration and tracking volume
        :type show_visualization: string
        """
        keypoints3D = np.array(self.keypoints3D)
        x_range = [np.min(keypoints3D[:,:,0]), np.max(keypoints3D[:,:,0])]
        y_range = [np.min(keypoints3D[:,:,1]), np.max(keypoints3D[:,:,1])]
        z_range = [np.min(keypoints3D[:,:,2]), np.max(keypoints3D[:,:,2])]
        tracking_area = np.array([x_range, y_range, z_range])
        x_cube_size_min = np.max(np.max(keypoints3D[:,:,0], axis = 1)-np.min(keypoints3D[:,:,0],axis = 1))
        y_cube_size_min = np.max(np.max(keypoints3D[:,:,1], axis = 1)-np.min(keypoints3D[:,:,1],axis = 1))
        z_cube_size_min = np.max(np.max(keypoints3D[:,:,2], axis = 1)-np.min(keypoints3D[:,:,2],axis = 1))
        min_cube_size = np.max([x_cube_size_min,y_cube_size_min,z_cube_size_min])
        final_bbox_suggestion = int(np.ceil((min_cube_size*1.1)/8)*8)
        resolution_suggestion = int(np.floor((final_bbox_suggestion/100.)))

        x_grid_size = x_range[1]-x_range[0]
        y_grid_size = y_range[1]-y_range[0]
        z_grid_size = z_range[1]-z_range[0]
        margin = 0.5   #this should be replaced by something dependent on final_bbox_suggestion
        grid_x_suggestion = [int(np.ceil((x_range[0]-x_grid_size*margin)/resolution_suggestion)*resolution_suggestion),
                             int(np.ceil((x_range[1]+x_grid_size*margin)/resolution_suggestion)*resolution_suggestion)]
        grid_y_suggestion = [int(np.ceil((y_range[0]-y_grid_size*margin)/resolution_suggestion)*resolution_suggestion),
                             int(np.ceil((y_range[1]+y_grid_size*margin)/resolution_suggestion)*resolution_suggestion)]
        grid_z_suggestion = [int(np.ceil((z_range[0]-z_grid_size*margin)/resolution_suggestion)*resolution_suggestion),
                             int(np.ceil((z_range[1]+z_grid_size*margin)/resolution_suggestion)*resolution_suggestion)]

        suggested_parameters = {
            'bbox': final_bbox_suggestion,
            'resolution': resolution_suggestion,
            'grid_x': grid_x_suggestion,
            'grid_y': grid_y_suggestion,
            'grid_z': grid_z_suggestion,
        }

        if show_visualization:
            figure = plt.figure()
            axes = figure.gca(projection='3d')
            visualizer = SetupVisualizer('Camera_T', self.root_dir, self.coco.dataset['calibration']['intrinsics'], self.coco.dataset['calibration']['extrinsics'])
            visualizer.plot_cameras(axes)
            visualizer.plot_tracking_area(tracking_area, axes)
            visualizer.plot_datapoints(self.keypoints3D[0], axes)
            plt.show()

        return suggested_parameters

    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        margin = 20 #Margin between pictures in pixels
        w = 4 # Width of the matrix (nb of images)
        h = 3 # Height of the matrix (nb of images)
        n = w*h
        out_size = (1000,1000)

        x,y,z=zip(*sample[1])
        center3D = np.array([(max(x)+min(x))/2., (max(y)+min(y))/2., (max(z)+min(z))/2.])
        center3D = (center3D-np.mod(center3D, 5)).astype(np.int)
        #grid_size = lookup_subset.shape[0]
        xx,yy,zz = np.meshgrid(np.arange(52), np.arange(52), np.arange(52), indexing='ij')
        #heatmap3D = self.get_heatmap_value(lookup_subset,np.transpose(np.array(sample[3]), axes = [1,0,2,3]), xx,yy,zz)
        heatmap3D = sample[4]

        figure = plt.figure()
        axes = figure.gca(projection='3d')


        colors = [(0,0,1,max(0,3./2*c-0.5)) for c in np.linspace(0,1,100)]
        cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

        #axes.scatter(xx*self.cfg.VORTEX.GRID_SPACING+center3D[0]-100,yy*self.cfg.VORTEX.GRID_SPACING+center3D[1]-100,zz*self.cfg.VORTEX.GRID_SPACING+center3D[2]-100, c = heatmap3D[self.num_cameras,xx,yy,zz], cmap = cmapblue)
        points3D_hm = []
        for heatmap in heatmap3D:
            mean = [0,0,0]
            norm = 0
            for x in range(heatmap3D.shape[1]):
                for y in range(heatmap3D.shape[2]):
                    for z in range(heatmap3D.shape[3]):
                        if heatmap[x,y,z] > 128:
                            norm += heatmap[x,y,z]
                            mean[0] += ((x*self.cfg.VORTEX.GRID_SPACING+center3D[0])-100)*heatmap[x,y,z]
                            mean[1] += ((y*self.cfg.VORTEX.GRID_SPACING+center3D[1])-100)*heatmap[x,y,z]
                            mean[2] += ((z*self.cfg.VORTEX.GRID_SPACING+center3D[2])-100)*heatmap[x,y,z]
            points3D_hm.append(np.array(mean)/norm)


        c = ['r', 'r','r','r','b','b','b','b','g','g','g','g', 'orange', 'orange','orange','orange', 'y','y','y','y','purple', 'purple','purple']
        line_idxs = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [8,9], [9,10], [10,11], [12,13], [13,14], [14,15], [16,17], [17,18], [18,19], [3,7], [7,11], [11,15], [3,21], [7,21],[11,22], [15,22],[21,22], [18,15], [19,22]]

        keypoins3D = sample[1]
        for i, point in enumerate(keypoins3D):
            #print ("Classic:", i, point)
            if i != 20:
                axes.scatter(point[0], point[1], point[2], color = "gray")
        for line in line_idxs:
            axes.plot([keypoins3D[line[0]][0], keypoins3D[line[1]][0]], [keypoins3D[line[0]][1], keypoins3D[line[1]][1]], [keypoins3D[line[0]][2], keypoins3D[line[1]][2]], c = 'gray')

        #c = ['r', 'r','r','r','b','b','b','b','g','g','g','g', 'orange', 'orange','orange','orange', 'y','y','y','y','grey', 'grey','grey']
        #for i, point in enumerate(points3D_hm):
            #axes.scatter(point[0], point[1], point[2], color = c[i])
            #print ("HM:", i, points3D_hm[i])

        #axes.scatter(xx*self.cfg.VORTEX.GRID_SPACING+center3D[0]-100,yy*self.cfg.VORTEX.GRID_SPACING+center3D[1]-100,zz*self.cfg.VORTEX.GRID_SPACING+center3D[2]-100)
        #self.reproTool.renderPoints(sample[4], axes = axes, show_cameras = False)

        cube = np.array(list(itertools.product(*zip([-50,-50,-50],[50,50,50]))))
        cube = cube + center3D
        for corner in cube:
            axes.scatter(corner[0],corner[1],corner[2], c = 'magenta', s = 20)

        vertices = [[0,1], [0,2], [2,3], [1,3], [4,5], [4,6], [6,7], [5,7], [0,4], [1,5], [2,6], [3,7]]
        for vertex in vertices:
            axes.plot([cube[vertex[0]][0],cube[vertex[1]][0]],[cube[vertex[0]][1],cube[vertex[1]][1]],[cube[vertex[0]][2],cube[vertex[1]][2]],c = 'magenta')

        plt.show()



class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        return [(sample[0].astype(np.float32) - self.mean) / self.std, sample[1], sample[2], sample[3], sample[4]]



if __name__ == "__main__":
    from lib.config.project_manager import ProjectManager

    project = ProjectManager()
    project.load('12Cam')


    cfg = project.get_cfg()
    idx = 0
    training_set = VortexDataset3D(cfg = cfg, set='val')
    print (len(training_set))
    frameNames = []
    for i in range(len(training_set.image_ids)):
        image_info = training_set.coco.loadImgs(training_set.image_ids[i])[0]
        print (image_info['file_name'])
        frameNames.append(image_info['file_name'])
    print (len(frameNames))

    import csv
    with open("framesNames.csv", 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
     for name in frameNames:
         wr.writerow([name])

    #training_set.get_dataset_config(show_visualization = True)
    for i in range(0,121):
        training_set.visualize_sample(i)
    val_frame_numbers = []
    print (len(training_set.image_ids))
    #for i in range(len(training_set.image_ids)):
        #image_info = training_set.coco.loadImgs(training_set.image_ids[i])[0]
        #path = os.path.join(training_set.root_dir, training_set.set_name, image_info['file_name'])
        #frameset_ids = training_set.coco.dataset['framesets'][image_info['file_name'].split('/')[-1]]
        #val_frame_numbers.append(path.split(".")[0].split("_")[3])
        #print (path.split(".")[0].split("_")[3])
