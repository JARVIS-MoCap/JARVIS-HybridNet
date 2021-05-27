import os,sys,inspect
import torch
import numpy as np
import yaml
import cv2

import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage, BoundingBox, BoundingBoxesOnImage


from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms

from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
#from core.hrnet.target_generators import HeatmapGenerator
#from core.hrnet.target_generators import JointsGenerator

class CocoDataset(Dataset):
    def __init__(self, cfg, set='train', mode = 'cropping'):
        self.cfg = cfg
        self.root_dir = cfg.DATASET.DATASET_DIR
        self.set_name = set
        self.mode = mode
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.num_keypoints = []
        for category in self.coco.dataset['categories']:
            self.num_keypoints.append(category['num_keypoints'])
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

        #self.heatmap_generator = [
        #    HeatmapGenerator(
        #        320, output_size, cfg.DATASET.HRNET.NUM_JOINTS, cfg.DATASET.HRNET.SIGMA
        #    ) for output_size in cfg.DATASET.HRNET.OUTPUT_SIZE
        #]
        #self.joints_generator = [
        #    JointsGenerator(
        #        cfg.DATASET.HRNET.MAX_NUM_PEOPLE,
        #        cfg.DATASET.HRNET.NUM_JOINTS,
        #        output_size,
        #        cfg.DATASET.HRNET.TAG_PER_JOINT
        #    ) for output_size in cfg.DATASET.HRNET.OUTPUT_SIZE
        #]
        #self.num_scales = self._init_check(self.heatmap_generator, self.joints_generator)

        if mode == 'cropping':
            width, height = cfg.DATASET.ORIGINAL_IMG_SIZE
            self.scale = cfg.DATASET.EFFICIENTDET.IMG_SIZE / max(height, width)
            self.augpipe = iaa.Sequential([
                iaa.Resize(self.scale,  interpolation='linear'),
                iaa.Affine(rotate=(-25,25),scale=(0.7,1.75)),
                iaa.PadToFixedSize(256,256)],
                random_order=False)
        elif mode == 'keypoints':
            self.scale = cfg.DATASET.HRNET.IMG_SIZE / cfg.DATASET.HRNET.BOUNDING_BOX_SIZE
            self.augpipe = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                iaa.LinearContrast((0.7, 1)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02), per_channel=0.6),
                iaa.Multiply((0.7, 1.5)),
                iaa.Fliplr(0.5),
                iaa.Sometimes(0.3, iaa.Multiply((0.8, 1.2), per_channel=1.0)),
                iaa.Resize(self.scale,  interpolation='linear'),
                iaa.Affine(rotate=(-60,60),scale=(0.8,1.25))],
                #iaa.PadToFixedSize(256,256)],
                random_order=False)
        elif mode == '3D':
            self.scale = 1.0
            self.reproTool = ReprojectionTool('T', self.root_dir, self.coco.dataset['calibration']['intrinsics'], self.coco.dataset['calibration']['extrinsics'])
            #self.reproTool.renderPoints([], True)
            self.image_ids_all = self.image_ids
            self.image_ids = [self.coco.dataset['framesets'][set][0] for set in self.coco.dataset['framesets']]

        self.transform = transforms.Compose([Normalizer(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD, scale = self.scale, mode=mode)])



    def load_classes(self):
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


    def __getitem__(self, idx):
        if self.mode == 'cropping':
            img = self.load_image(idx)
            bboxs, keypoints = self.load_annotations(idx)
            bboxs_iaa = BoundingBoxesOnImage([BoundingBox(x1=bboxs[0][0], y1=bboxs[0][1], x2=bboxs[0][2], y2=bboxs[0][3])], shape=(img.shape[0],img.shape[1],3))
            img, bboxs_aug = self.augpipe(image=img, bounding_boxes=bboxs_iaa)
            bboxs[0][0] = bboxs_aug[0].x1
            bboxs[0][1] = bboxs_aug[0].y1
            bboxs[0][2] = bboxs_aug[0].x2
            bboxs[0][3] = bboxs_aug[0].y2
            sample = {'img': img, 'bboxs': bboxs}

        elif self.mode == 'keypoints':
            img = self.load_image(idx)
            bboxs, keypoints = self.load_annotations(idx)
            bbox_hw = int(self.cfg.DATASET.HRNET.BOUNDING_BOX_SIZE/2)
            center_y = min(max(bbox_hw, int((bboxs[0][1]+int(bboxs[0][3]))/2)), 1024-bbox_hw)
            center_x = min(max(bbox_hw, int((bboxs[0][0]+int(bboxs[0][2]))/2)), 1280-bbox_hw)
            img = img[center_y-bbox_hw:center_y+bbox_hw, center_x-bbox_hw:center_x+bbox_hw, :]
            for i in range(0, keypoints.shape[1],3):
                keypoints[0][i] += -center_x+bbox_hw
                keypoints[0][i+1] += -center_y+bbox_hw
            if self.set_name == 'train':
                keypoints_iaa = KeypointsOnImage([Keypoint(x=keypoints[0][i], y=keypoints[0][i+1]) for i in range(0,len(keypoints[0]),3)],
                                                 shape=(self.cfg.DATASET.HRNET.BOUNDING_BOX_SIZE,self.cfg.DATASET.HRNET.BOUNDING_BOX_SIZE,3))
                img, keypoints_aug = self.augpipe(image=img, keypoints = keypoints_iaa)
                for i,point in enumerate(keypoints_aug.keypoints):
                    keypoints[0][i*3] = point.x
                    keypoints[0][i*3+1] = point.y

            joints = np.zeros((1,self.cfg.DATASET.HRNET.NUM_JOINTS, 3))
            joints[0, :self.cfg.DATASET.HRNET.NUM_JOINTS, :3] = np.array(keypoints[0]).reshape([-1, 3])
            joints_list = [joints.copy() for _ in range(self.num_scales)]
            target_list = list()
            for scale_id in range(self.num_scales):
                target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
                joints_t = self.joints_generator[scale_id](joints_list[scale_id])
                target_list.append(target_t.astype(np.float32))
                joints_list[scale_id] = joints_t.astype(np.int32)
            sample = [img, target_list, keypoints]

        if self.transform:
            sample = self.transform(sample)
        return sample


    def load_image(self, image_index, is_id = False):
        if is_id:
            image_info = self.coco.loadImgs(image_index)[0]
        else:
            image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        return img


    def load_annotations(self, image_index, is_id = False):
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


    def visualize_sample(self, idx):
        sample = self.__getitem__(idx)
        if self.mode == 'keypoints':
            img = (sample[0].numpy()*self.cfg.DATASET.STD+self.cfg.DATASET.MEAN)
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
            img = (sample['img'].numpy()*self.cfg.DATASET.STD+self.cfg.DATASET.MEAN)
            img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
            bboxs = sample['bboxs'].numpy()
            for i,bbox in enumerate(bboxs):
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
                cv2.putText(img, '{}'.format(self.labels[int(bbox[4])]),
                            (int(bbox[0]), int(bbox[1]) +8), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255), 1)
            cv2.imshow('frame', img)
            cv2.waitKey(0)


    def _init_check(self, heatmap_generator, joints_generator):
        assert isinstance(heatmap_generator, (list, tuple)), 'heatmap_generator should be a list or tuple'
        assert isinstance(joints_generator, (list, tuple)), 'joints_generator should be a list or tuple'
        assert len(heatmap_generator) == len(joints_generator), \
            'heatmap_generator and joints_generator should have same length,'\
            'got {} vs {}.'.format(
                len(heatmap_generator), len(joints_generator)
            )
        return len(heatmap_generator)


def collater_bbox(data):
    imgs = [s['img'] for s in data]
    annots = [s['bboxs'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

def collater_keypoints(data):
    imgs = [s[0] for s in data]
    heatmaps = [torch.from_numpy(np.array([s[1][i] for s in data])) for i in range(2)]
    scales = [s[2] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    #heatmaps = torch.from_numpy(np.stack(heatmaps, axis=0))

    imgs = imgs.permute(0, 3, 1, 2)

    return [imgs, heatmaps, scales]


class Normalizer(object):
    def __init__(self, mean, std, scale, mode = 'cropping'):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
        self.mode = mode
        self.scale = scale

    def __call__(self, sample):
        if self.mode == 'cropping':
            image, bboxs = sample['img'], sample['bboxs']
            return {'img': torch.from_numpy((image.astype(np.float32) - self.mean) / self.std).to(torch.float32), 'bboxs': torch.from_numpy(bboxs), 'scale': self.scale}

        elif self.mode == 'keypoints':
            image, heatmaps = sample[0], sample[1]
            keypoints = sample[2]
            return [torch.from_numpy((image.astype(np.float32) - self.mean) / self.std).to(torch.float32), heatmaps, keypoints,self.scale]


if __name__ == "__main__":
    from config import cfg
    training_set = CocoDataset(cfg = cfg, set='val', mode='cropping')
    print (len(training_set.image_ids))
    for i in range(0,1000,12):
        training_set.visualize_sample(i)
    #training_set.__getitem__(0)
