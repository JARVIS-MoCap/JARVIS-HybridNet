"""
dataset2D.py
============
Vortex 2D dataset loader.
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

from lib.dataset.datasetBase import VortexBaseDataset


class VortexDataset2D(VortexBaseDataset):
    """
    Dataset Class to load 2D datasets in the VoRTEx dataset format, inherits from
    VortexBaseDataset class. See HERE for more details.

    :param cfg: handle of the global configuration
    :param set: specifies wether to load training ('train') or validation ('val') split.
                Augmentation will only be applied to training split.
    :type set: string
    :param mode: specifies wether bounding box annotations ('cropping') or keypoint
                 annotations ('keypoints') will be loaded.
    :type mode: string
    """
    def __init__(self, cfg, set='train', mode = 'cropping'):
        dataset_name = cfg.DATASET.DATASET_2D
        super().__init__(cfg, dataset_name,set)
        self.mode = mode
        assert cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE % 64 == 0, "Bounding Box size has to be divisible by 64!"
        self.heatmap_generator = [
            HeatmapGenerator(
                cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE, output_size, self.num_keypoints[0])  \
                    for output_size in [int(cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE/4), int(cfg.EFFICIENTTRACK.BOUNDING_BOX_SIZE/2)]
        ]
        self._build_augpipe()
        self.transform = transforms.Compose([Normalizer(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD, mode=mode)])


    def _build_augpipe(self):
        augmentors = []
        if self.mode == 'cropping':
            img = self._load_image(0)
            width, height = img.shape[1], img.shape[0]
            scale = self.cfg.EFFICIENTDET.IMG_SIZE / max(height, width)
            cfg = self.cfg.EFFICIENTDET.AUGMENTATION
            augmentors.append(iaa.Resize(scale,  interpolation='linear'))

        elif self.mode == 'keypoints':
            cfg = self.cfg.EFFICIENTTRACK.AUGMENTATION

        if not (self.mode == 'cropping' and self.set_name == 'val'):
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
        if self.mode == 'cropping':
            augmentors.append(iaa.PadToFixedSize(self.cfg.EFFICIENTDET.IMG_SIZE,self.cfg.EFFICIENTDET.IMG_SIZE))

        self.augpipe = iaa.Sequential(augmentors, random_order = False)


    def __getitem__(self, idx):
        if self.mode == 'cropping':
            img = self._load_image(idx)
            bboxs, keypoints = self._load_annotations(idx)
            for i,bbox in enumerate(bboxs):
                bboxs_iaa = BoundingBoxesOnImage([BoundingBox(x1=bboxs[i,0], y1=bboxs[i,1], x2=bboxs[i,2], y2=bboxs[i,3])], shape=(img.shape[0],img.shape[1],3))
                img, bboxs_aug = self.augpipe(image=img, bounding_boxes=bboxs_iaa)
                bboxs[i,0] = bboxs_aug[0].x1
                bboxs[i,1] = bboxs_aug[0].y1
                bboxs[i,2] = bboxs_aug[0].x2
                bboxs[i,3] = bboxs_aug[0].y2
            sample = [img, bboxs]

        elif self.mode == 'keypoints':
            img = self._load_image(idx)
            bboxs, keypoints = self._load_annotations(idx)
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
        print (path)


        final_bbox_suggestion = int(np.ceil((bbox_min_size*1.02)/64)*64)
        return final_bbox_suggestion


    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        if self.mode == 'keypoints':
            img = (sample[0]*self.cfg.DATASET.STD+self.cfg.DATASET.MEAN)
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
            heatmaps = sample[1]
            img = cv2.resize(img*255, (heatmaps[1][0].shape[0], heatmaps[1][0].shape[1])).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmaps[1][0].astype(np.uint8), cv2.COLORMAP_JET)
            for i in range(1,heatmaps[1].shape[0]):
                colored_heatmap = colored_heatmap + cv2.applyColorMap(heatmaps[1][i].astype(np.uint8), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img,1.0,colored_heatmap,0.4,0)
            cv2.imshow('frame', img)
            cv2.waitKey(0)

        elif self.mode == 'cropping':
            img = (sample[0].numpy()*self.cfg.DATASET.STD+self.cfg.DATASET.MEAN)
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
            bboxs = sample[1].numpy()
            for i,bbox in enumerate(bboxs):
                if bbox[4] != -1:
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
                    cv2.putText(img, '{}'.format(self.labels[int(bbox[4])]),
                                (int(bbox[0]), int(bbox[1]) +8), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                (0, 0, 255), 1)
            img = cv2.resize(img, (512,512));
            cv2.imshow('frame', img)
            cv2.waitKey(0)

class Normalizer(object):
    def __init__(self, mean, std, mode = 'cropping'):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'cropping':
            image, bboxs = sample[0], sample[1]
            return [torch.from_numpy((image.astype(np.float32) - self.mean) / self.std).to(torch.float32), torch.from_numpy(bboxs)]

        elif self.mode == 'keypoints':
            image, heatmaps = sample[0], sample[1]
            keypoints = sample[2]
            return [(image.astype(np.float32) - self.mean) / self.std, heatmaps, keypoints]


class HeatmapGenerator():
    def __init__(self, original_res, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        self.scale_factor = float(output_res)/float(original_res)
        if sigma < 0:
            sigma = 2*self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = 255.0*np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]*self.scale_factor), int(pt[1]*self.scale_factor)
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

if __name__ == "__main__":
    from lib.config.project_manager import ProjectManager
    project = ProjectManager()
    project.load('TestNew')
    cfg = project.get_cfg()
    print (cfg.DATASET.DATASET_2D)

    training_set = VortexDataset2D(cfg = cfg, set='train', mode='keypoints')
    print (len(training_set.image_ids))
    for i in range(0,len(training_set.image_ids),10):
        training_set.visualize_sample(i)
    #training_set.__getitem__(0)
