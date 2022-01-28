"""
datasetBase.py
==============
HybridNet dataset loader base class.
Strongly inspired and partially taken from pycocotools
(https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
"""

import os,sys,inspect
import torch
import numpy as np
import cv2
import json
from collections import defaultdict

from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(
                              inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class BaseDataset(Dataset):
    """
    Dataset Class to load datasets in the HybridNet dataset format.

    :param cfg: handle of the global configuration
    :param dataset_name: name of the dataset to be loaded
    :type dataset_name: string
    :param set: specifies wether to load training ('train') or
                validation ('val') split. Augmentation will only be applied
                to training split.
    :type set: string
    """
    def __init__(self, cfg, dataset_name,set='train'):
        self.cfg = cfg
        self.root_dir = os.path.join(cfg.DATASET.DATASET_ROOT_DIR, dataset_name)
        self.set_name = set

        dataset_file = open(os.path.join(self.root_dir, 'annotations',
                    'instances_' + self.set_name + '.json'))
        self.dataset = json.load(dataset_file)

        self.num_keypoints = []
        for category in self.dataset['categories']:
            self.num_keypoints.append(category['num_keypoints'])
        self.image_ids = [img["id"] for img in self.dataset["images"]]

        self.annotations,self.categories,self.imgs = dict(),dict(),dict()
        self.imgToAnns = defaultdict(list)
        self.createIndex()


    def createIndex(self):
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.imgToAnns[ann['image_id']].append(ann)
                self.annotations[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                self.categories[cat['id']] = cat


    def __len__(self):
        return len(self.image_ids)


    def _load_image(self, image_index, is_id = True):
        if is_id:
            file_name = self.imgs[image_index]['file_name']
        else:
            file_name = self.imgs[self.image_ids[image_index]]['file_name']
        path = os.path.join(self.root_dir, self.set_name, file_name)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        return img


    def _load_annotations(self, image_index, is_id = False):
        if is_id:
            annotations_ids = [ann['id'] for ann in self.imgToAnns[image_index]]
        else:
            annotations_ids = [ann['id'] for ann in self.imgToAnns[self.image_ids[image_index]]]
        annotations = np.zeros((0, 5))
        keypoints = np.zeros((0,self.num_keypoints[0]*3))

        if len(annotations_ids) == 0:
            annotations = np.zeros((1, 5))
            annotations[0][4] = -1
            keypoints = np.zeros((1,self.num_keypoints[0]*3))
            return annotations, keypoints

        coco_annotations = self._loadAnns(annotations_ids)

        for idx, a in enumerate(coco_annotations):
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            for i in range(len(annotation)):
                annotation[i] /= 1
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)
            keypoint = np.array(a['keypoints']).reshape(1,
                        self.num_keypoints[0]*3)
            keypoints =np.append(keypoints, keypoint, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations, keypoints


    def _loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        return [self.annotations[id] for id in ids]
