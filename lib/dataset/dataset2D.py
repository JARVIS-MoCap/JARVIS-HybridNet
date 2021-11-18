"""
dataset2D.py
============
HybridNet 2D dataset loader.
"""

import os,sys,inspect
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from lib.dataset.datasetBase import BaseDataset


class Dataset2D(BaseDataset):
    """
    Dataset Class to load 2D datasets in the HybridNet dataset format, inherits from
    BaseDataset class. See HERE for more details.

    :param cfg: handle of the global configuration
    :param set: specifies wether to load training ('train') or validation ('val') split.
                Augmentation will only be applied to training split.
    :type set: string
    :param mode: specifies wether center of mass ('center') or keypoint
                 annotations ('keypoints') will be loaded.
    :type mode: string
    """
    def __init__(self, cfg, set='train', mode = 'center'):
        dataset_name = cfg.DATASET.DATASET_2D
        super().__init__(cfg, dataset_name,set)
        self.mode = mode
        assert cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE % 64 == 0, "Bounding Box size has to be divisible by 64!"
        if self.mode == "center":
            cfg.EFFICIENTTRACK.NUM_JOINTS = 1
            img = self._load_image(0)
            width, height = img.shape[1], img.shape[0]
            scale = self.cfg.EFFICIENTDET.IMG_SIZE / max(height, width)
            self.heatmap_generator = [
                HeatmapGenerator(
                    [height*scale,width*scale], output_size, 1, sigma  = -2)  \
                        for output_size in [[int(height*scale/4),int(width*scale/4)], [int(height*scale/2),int(width*scale/2)]]
            ]
        elif self.mode == "keypoints":
            cfg.EFFICIENTTRACK.NUM_JOINTS = self.num_keypoints[0]
            self.heatmap_generator = [
                HeatmapGenerator(
                    [cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE,cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE] , [output_size,output_size], self.num_keypoints[0])  \
                        for output_size in [int(cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE/4), int(cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE/2)]
            ]
        self._build_augpipe()
        self.transform = transforms.Compose([Normalizer(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)])


    def _build_augpipe(self):
        augmentors = []
        if self.mode == 'center':
            img = self._load_image(0)
            width, height = img.shape[1], img.shape[0]
            scale = self.cfg.EFFICIENTDET.IMG_SIZE / max(height, width)
            cfg = self.cfg.EFFICIENTDET.AUGMENTATION
            augmentors.append(iaa.Resize(scale,  interpolation='linear'))

        elif self.mode == 'keypoints':
            cfg = self.cfg.EFFICIENTTRACK.AUGMENTATION

        if not (self.mode == 'center' and self.set_name == 'val'):
            if cfg.COLOR_MANIPULATION.ENABLED:
                cman = cfg.COLOR_MANIPULATION
                augmentors.append(iaa.Sometimes(cman.GAUSSIAN_BLUR.PROBABILITY,
                                  iaa.GaussianBlur(sigma=cman.GAUSSIAN_BLUR.SIGMA)))
                augmentors.append(iaa.AdditiveGaussianNoise(loc = 0, scale = cman.GAUSSIAN_NOISE.SCALE,
                                  per_channel = cman.GAUSSIAN_NOISE.PER_CHANNEL_PROBABILITY))
                augmentors.append(iaa.Sometimes(cman.LINEAR_CONTRAST.PROBABILITY,
                                  iaa.LinearContrast(cman.LINEAR_CONTRAST.SCALE)))
                augmentors.append(iaa.Sometimes(cman.MULTIPLY.PROBABILITY,
                                  iaa.Multiply(cman.MULTIPLY.SCALE)))
                augmentors.append(iaa.Sometimes(cman.PER_CHANNEL_MULTIPLY.PROBABILITY,
                                  iaa.Multiply(cman.PER_CHANNEL_MULTIPLY.SCALE,
                                  per_channel=cman.PER_CHANNEL_MULTIPLY.PER_CHANNEL_PROBABILITY)))

            augmentors.append(iaa.Fliplr(cfg.MIRROR.PROBABILITY))
            augmentors.append(iaa.Sometimes(cfg.AFFINE_TRANSFORM.PROBABILITY,
                              iaa.Affine(rotate=cfg.AFFINE_TRANSFORM.ROTATION_RANGE,
                              scale=cfg.AFFINE_TRANSFORM.SCALE_RANGE)))

        self.augpipe = iaa.Sequential(augmentors, random_order = False)


    def __getitem__(self, idx):
        img = self._load_image(idx)
        bboxs, keypoints = self._load_annotations(idx)
        if self.mode == 'center':
            bbox_hw = int(self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE/2)
            center_y = min(max(bbox_hw, int((bboxs[0][1]+int(bboxs[0][3]))/2)), img.shape[0]-bbox_hw)
            center_x = min(max(bbox_hw, int((bboxs[0][0]+int(bboxs[0][2]))/2)), img.shape[1]-bbox_hw)
            keypoints_masked = np.array(keypoints[0]).reshape(-1,3)
            keypoints_masked = np.ma.MaskedArray(keypoints_masked, mask=(np.ones_like(keypoints_masked)*(keypoints_masked[:,2]==0).reshape(-1,1)))
            center_median = np.ma.median(keypoints_masked, axis = 0)
            if bboxs[0][4]  != -1:
                center = np.array([[center_median[0],center_median[1],1]])
            else:
                center = np.array([[0.0,0.0,1.0]])
            keypoints_iaa = KeypointsOnImage([Keypoint(x=center[0][0], y=center[0][1])],
                                             shape=(1024,1280,3))
            img, keypoints_aug = self.augpipe(image=img, keypoints = keypoints_iaa)
            center[0][0] = keypoints_aug[0].x
            center[0][1] = keypoints_aug[0].y

            joints = np.zeros((1,1, 3))
            joints[0, :1, :3] = center.reshape([-1, 3])
            joints_list = [[],[]]
            if bboxs[0][4]  != -1:
                joints_list = [joints.copy() for _ in range(2)]
            target_list = list()
            for scale_id in range(2):
                target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
                target_list.append(target_t.astype(np.float32))
            sample = [img, target_list, center]
            return self.transform(sample)

        elif self.mode == 'keypoints':
            bbox_hw = int(self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE/2)
            center_y = min(max(bbox_hw, int((bboxs[0][1]+int(bboxs[0][3]))/2)), img.shape[0]-bbox_hw)
            center_x = min(max(bbox_hw, int((bboxs[0][0]+int(bboxs[0][2]))/2)), img.shape[1]-bbox_hw)
            img = img[center_y-bbox_hw:center_y+bbox_hw, center_x-bbox_hw:center_x+bbox_hw, :]
            for i in range(0, keypoints.shape[1],3):
                keypoints[0,i] += -center_x+bbox_hw
                keypoints[0,i+1] += -center_y+bbox_hw
            if self.set_name == 'train':
                keypoints_iaa = KeypointsOnImage([Keypoint(x=keypoints[0][i], y=keypoints[0][i+1]) for i in range(0,len(keypoints[0]),3)],
                                                 shape=(self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE,self.cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE,3))
                img, keypoints_aug = self.augpipe(image=img, keypoints = keypoints_iaa)
                for i,point in enumerate(keypoints_aug.keypoints):
                    keypoints[0,i*3] = point.x
                    keypoints[0,i*3+1] = point.y

            joints = np.zeros((1,self.num_keypoints[0], 3))
            joints[0, :self.num_keypoints[0], :3] = np.array(keypoints[0]).reshape([-1, 3])
            joints_list = [joints.copy() for _ in range(2)]
            target_list = list()
            for scale_id in range(2):
                target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
                target_list.append(target_t.astype(np.float32))
            sample = [img, target_list, keypoints]
        return self.transform(sample)

    def __len__(self):
        return len(self.image_ids)


    def get_dataset_config(self):
        """
        Get the recommended configuration for the 2D Dataset. Recommendations are
        computed by analyzing the trainingset, if it is not representative of the data
        you plan to analyze, the parameters might need to be adjusted manually
        """
        bboxs = []
        for id in self.image_ids:
            bbox, _ = self._load_annotations(id)
            if len(bbox) != 0:
                bboxs.append(bbox)
        bboxs = np.array(bboxs)
        x_sizes = bboxs[:,0,2]-bboxs[:,0,0]
        y_sizes = bboxs[:,0,3]-bboxs[:,0,1]
        bbox_min_size = np.max([np.max(x_sizes), np.max(y_sizes)])
        ind = np.argmax(x_sizes)
        image_info = self.coco.loadImgs(self.image_ids[self.image_ids[ind]])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])

        final_bbox_suggestion = int(np.ceil((bbox_min_size*1.02)/64)*64)
        return final_bbox_suggestion


    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        if self.mode == 'keypoints' or self.mode == "center":
            img = (sample[0]*self.cfg.DATASET.STD+self.cfg.DATASET.MEAN)
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
            heatmaps = sample[1]
            img = cv2.resize(img*255, (heatmaps[1][0].shape[1], heatmaps[1][0].shape[0])).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmaps[1][0].astype(np.uint8), cv2.COLORMAP_JET)
            for i in range(1,heatmaps[1].shape[0]):
                colored_heatmap = colored_heatmap + cv2.applyColorMap(heatmaps[1][i].astype(np.uint8), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img,1.0,colored_heatmap,0.4,0)
            img = cv2.resize(img, (640,512))
            cv2.imshow('frame', img)
            cv2.waitKey(0)


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, heatmaps = sample[0], sample[1]
        keypoints = sample[2]
        return [(image.astype(np.float32) - self.mean) / self.std, heatmaps, keypoints]


class HeatmapGenerator():
    def __init__(self, original_res, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        self.scale_factor = float(output_res[0])/float(original_res[0])
        if sigma == -1:
            sigma = 2*self.output_res[0]/64
        elif sigma == -2:
            sigma = 1*self.output_res[0]/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = 255.0*np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res[0], self.output_res[1]),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]*self.scale_factor), int(pt[1]*self.scale_factor)
                    if x < 0 or y < 0 or \
                       x >= self.output_res[1] or y >= self.output_res[0]:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res[1]) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res[0]) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res[1])
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res[0])
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

if __name__ == "__main__":
    from lib.config.project_manager import ProjectManager
    project = ProjectManager()
    project.load('Ralph_Center_Test')
    cfg = project.get_cfg()
    print (cfg.DATASET.DATASET_2D)

    training_set = Dataset2D(cfg = cfg, set='train', mode='center')
    print (len(training_set.image_ids))
    for i in range(0,len(training_set.image_ids),10):
        training_set.visualize_sample(i)
    #training_set.__getitem__(0)
