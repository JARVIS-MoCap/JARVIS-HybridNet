"""
datasetBase.py
========
Vortex dataset loader base class.
"""


import os,sys,inspect
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset
from pycocotools.coco import COCO

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class VortexBaseDataset(Dataset):
    """
    Dataset Class to load datasets in the VoRTEx dataset format. See HERE for more details.

    :param cfg: handle of the global configuration
    :param set: specifies wether to load training ('train') or validation ('val') split.
                Augmentation will only be applied to training split.
    :type set: string
    :param mode: specifies wether bounding box annotations ('cropping') or keypoint
                 annotations ('keypoints') will be loaded.
    """
    def __init__(self, cfg, set='train'):
        self.cfg = cfg
        self.root_dir = cfg.DATASET.DATASET_DIR
        self.set_name = set
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.num_keypoints = []
        for category in self.coco.dataset['categories']:
            self.num_keypoints.append(category['num_keypoints'])
        self.image_ids = self.coco.getImgIds()
        self._load_classes()
        if self.cfg.DATASET.OBJ_LIST == None:
            self.cfg.DATASET.OBJ_LIST = [value for key, value in self.labels.items()]

    def _load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key


    def __len__(self):
        return len(self.image_ids)


    def _load_image(self, image_index, is_id = False):
        if is_id:
            image_info = self.coco.loadImgs(image_index)[0]
        else:
            image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        return img


    def _load_annotations(self, image_index, is_id = False):
        # get ground truth annotations
        if is_id:
            annotations_ids = self.coco.getAnnIds(imgIds=image_index, iscrowd=False)
        else:
            annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        keypoints = np.zeros((0,self.num_keypoints[0]*3))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            #if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            #    continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            for i in range(len(annotation)):
                annotation[i] /= 1
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)
            keypoint = np.array(a['keypoints']).reshape(1,69)
            keypoints =np.append(keypoints, keypoint, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations, keypoints
