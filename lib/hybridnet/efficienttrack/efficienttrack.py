"""
efficienttrack.py
=================
EfficientTrack convenience class, can be used to train and troublshoot the
EfficientTrack module.
"""

import os
import numpy as np
from tqdm.autonotebook import tqdm
import traceback
import cv2
import time
import csv
import itertools
import matplotlib


import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import EfficientTrackBackbone
from .loss import HeatmapLoss
import lib.hybridnet.efficienttrack.utils as utils
from lib.logger.logger import NetLogger, AverageMeter
import lib.hybridnet.efficienttrack.darkpose as darkpose

import warnings
#Filter out weird pytorch floordiv deprecation warning, don't know where it's
#coming from so can't really fix it
warnings.filterwarnings("ignore", category=UserWarning)

class EfficientTrack:
    """
    EfficientTrack convenience class, enables easy training and inference with
    using the EfficientTrack Module.

    :param mode: Select wether the network is loaded in training or inference
                 mode
    :type mode: string
    :param cfg: Handle for the global configuration structure
    :param weights: Path to parameter savefile to be loaded
    :type weights: string, optional
    """
    def __init__(self, mode, cfg, weights = None, run_name = None):
        self.mode = mode
        self.main_cfg = cfg
        if mode == 'CenterDetect' or mode == 'CenterDetectInference':
            self.cfg = self.main_cfg.CENTERDETECT
        else:
            self.cfg = self.main_cfg.KEYPOINTDETECT
        self.model = EfficientTrackBackbone(self.cfg,
                    compound_coef=self.cfg.COMPOUND_COEF,
                    output_channels = self.cfg.NUM_JOINTS)

        if mode  == 'KeypointDetect' or mode == 'CenterDetect':
            if run_name == None:
                run_name = "Run_" + time.strftime("%Y%m%d-%H%M%S")

            self.model_savepath = os.path.join(self.main_cfg.savePaths[mode],
                        run_name)
            os.makedirs(self.model_savepath, exist_ok=True)
            self.logger = NetLogger(os.path.join(self.main_cfg.logPaths[mode],
                        run_name))

            self.lossMeter = AverageMeter()
            self.accuracyMeter = AverageMeter()

            self.load_weights(weights)

            self.criterion = HeatmapLoss(self.cfg, self.mode)
            self.model = self.model.cuda()


            if self.cfg.OPTIMIZER == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(),
                            self.cfg.LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                            self.cfg.LEARNING_RATE, momentum=0.9, nesterov=True)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, patience=3, verbose=True,
                        min_lr=0.00005, factor = 0.2)



        elif mode == 'KeypointDetectInference' or 'CenterDetectInference':
            self.model.requires_grad_(False)
            self.model.load_state_dict(torch.load(weights))
            self.model.requires_grad_(False)
            self.model.eval()
            self.model = self.model.cuda()

        elif mode == 'export':
            self.model = EfficientTrackBackbone(
                        num_classes=len(self.cfg.DATASET.OBJ_LIST),
                        compound_coef=self.cfg.COMPOUND_COEF,
                        onnx_export = True)
            self.load_weights(weights)
            self.model = self.model.cuda()



    def load_weights(self, weights_path = None):
        if weights_path is not None:
            pretrained_dict = torch.load(weights_path)
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k in ['final_conv1.weight', 'final_conv2.weight', 'first_conv.pointwise_conv.bias', 'first_conv.gn.weight', 'first_conv.gn.bias', 'first_conv.pointwise_conv.weight']}
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k in ['final_conv1.weight', 'final_conv2.weight']}
            self.model.load_state_dict(pretrained_dict, strict=False)
            print(f'Successfully loaded weights: {os.path.basename(weights_path)}')
        else:
            print('Initializing weights...')
            utils.init_weights(self.model)


    def freeze_backbone(self):
        classname = self.model.__class__.__name__
        for ntl in ['EfficientNet', 'BiFPN']:
            if ntl in classname:
                for param in self.model.parameters():
                    param.requires_grad = False


    def train(self, training_set, validation_set, num_epochs, start_epoch = 0):
        """
        Function to train the network on a given dataset for a set number of
        epochs. Most of the training parameters can be set in the config file.

        :param training_set: training dataset
        :type training_generator: TODO
        :param val_generator: validation dataset
        :type validation_set: TODO
        :param num_epochs: Number of epochs the network is trained for
        :type num_epochs: int
        :param start_epoch: Initial epoch for the training, set this if training
            is continued from an earlier session
        """
        training_generator = DataLoader(
                    training_set,
                    batch_size = self.cfg.BATCH_SIZE,
                    shuffle = True,
                    num_workers =  self.main_cfg.DATALOADER_NUM_WORKERS,
                    pin_memory = True,
                    drop_last = True)

        val_generator = DataLoader(
                    validation_set,
                    batch_size = self.cfg.BATCH_SIZE,
                    shuffle = False,
                    num_workers =  self.main_cfg.DATALOADER_NUM_WORKERS,
                    pin_memory = True,
                    drop_last = True)

        epoch = start_epoch
        best_loss = 1e5
        best_epoch = 0
        self.model.train()
        if self.main_cfg.USE_MIXED_PRECISION:
            scaler = torch.cuda.amp.GradScaler()

        #self.model.requires_grad_(False)
        #self.model.final_conv1.requires_grad_(True)
        #self.model.final_conv2.requires_grad_(True)
        #self.model.first_conv.requires_grad_(True)

        for epoch in range(num_epochs):
            #if epoch == 0:
            #    self.model.requires_grad_(True)
            #    print ("Training whole model now.")
            progress_bar = tqdm(training_generator)
            for data in progress_bar:
                imgs = data[0].permute(0, 3, 1, 2).float()
                heatmaps = data[1]

                imgs = imgs.cuda()
                heatmaps = list(map(lambda x: x.cuda(non_blocking=True),
                            heatmaps))

                self.optimizer.zero_grad()
                if self.main_cfg.USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs)
                        heatmaps_losses = self.criterion(outputs, heatmaps)
                        loss = 0
                        for idx in range(2):
                            if heatmaps_losses[idx] is not None:
                                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                                loss = loss + heatmaps_loss
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    outputs = self.model(imgs)
                    heatmaps_losses = self.criterion(outputs, heatmaps)
                    loss = 0
                    for idx in range(2):
                        if heatmaps_losses[idx] is not None:
                            heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                            loss = loss + heatmaps_loss
                    loss.backward()
                    self.optimizer.step()

                self.lossMeter.update(loss.item())
                progress_bar.set_description(
                    'Epoch: {}/{}. Loss: {:.5f}'.format(
                        epoch, num_epochs, self.lossMeter.read()))


            self.logger.update_train_loss(self.lossMeter.read())
            self.scheduler.step(self.lossMeter.read())

            self.lossMeter.reset()

            if (epoch+1) % self.cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                self.save_checkpoint(f'EfficientTrack-d{self.cfg.COMPOUND_COEF}_{epoch+1}.pth')
                print('checkpoint...')
            if (epoch+1) % self.cfg.VAL_INTERVAL == 0:
                self.model.eval()
                for data in val_generator:
                    with torch.no_grad():
                        imgs = data[0].permute(0, 3, 1, 2).float()
                        heatmaps = data[1]
                        keypoints = np.array(data[2]).reshape(-1,
                                    self.cfg.NUM_JOINTS,3)[:,:,:2]

                        imgs = imgs.cuda()
                        heatmaps = list(map(lambda x: x.cuda(non_blocking=True),
                                    heatmaps))

                        if self.main_cfg.USE_MIXED_PRECISION:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(imgs)
                                heatmaps_losses = self.criterion(outputs,
                                            heatmaps)
                        else:
                            outputs = self.model(imgs)
                            heatmaps_losses = self.criterion(outputs,
                                        heatmaps)

                        loss = 0
                        for idx in range(2):
                            if heatmaps_losses[idx] is not None:
                                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                                loss = loss + heatmaps_loss

                    preds, maxvals = darkpose.get_final_preds(
                                outputs[1].clamp(0,255).detach().cpu().numpy(),
                                None)
                    mask = np.sum(keypoints,axis = 2)
                    masked = np.ma.masked_where(mask == 0,
                                np.linalg.norm((preds+0.5)*2-keypoints,
                                axis = 2))

                    self.lossMeter.update(loss.item())
                    self.accuracyMeter.update(np.mean(masked))

                print(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}. Acc: {:1.3f}'.format(
                        epoch, num_epochs, self.lossMeter.read(),
                        self.accuracyMeter.read()))

                self.logger.update_val_loss(self.lossMeter.read())
                self.logger.update_val_accuracy(self.accuracyMeter.read())
                self.lossMeter.reset()
                self.accuracyMeter.reset()

                if (loss + self.cfg.EARLY_STOPPING_MIN_DELTA < best_loss
                            and self.cfg.USE_EARLY_STOPPING):
                    best_loss = loss
                    best_epoch = epoch
                    self.save_checkpoint(f'EfficientTrack-d{self.cfg.COMPOUND_COEF}_{epoch+1}.pth')

                self.model.train()

                # Early stopping
                if (epoch - best_epoch > self.cfg.EARLY_STOPPING_PATIENCE > 0
                            and self.cfg.USE_EARLY_STOPPING):
                    print('[Info] Stop training at epoch {}. The lowest loss '
                          'achieved is {}'.format(epoch, best_loss))
                    break

    def save_checkpoint(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.model_savepath, name))

    def predictCenter(self, img):
        img_vis = ((img*self.main_cfg.DATASET.STD)+self.main_cfg.DATASET.MEAN)*255
        img_shape = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.reshape(1,3,img_shape[0],img_shape[1]).cuda()
        outputs = self.model(img)
        preds, maxvals = darkpose.get_final_preds(
                    outputs[1].clamp(0,255).detach().cpu().numpy(), None)
        for i,point in enumerate(preds[0]):
            if (maxvals[0][i]) > 10:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, (255,0,0), thickness=5)
            else:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, (100,0,0), thickness=5)

        heatmap = cv2.resize(outputs[1].clamp(0,255).detach().cpu().numpy()[0][0]/255.,
                    (img_shape[1],img_shape[0]), interpolation=cv2.cv2.INTER_NEAREST)
        return preds, maxvals, img_vis, outputs


    def predictKeypoints(self, img, colorPreset = None):
        img_vis = ((img*self.main_cfg.DATASET.STD)+self.main_cfg.DATASET.MEAN)*255
        img= torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.reshape(1,3,self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                              self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda()
        outputs = self.model(img)
        colors = []
        if isinstance(colorPreset, str):
            colors, _ = self.get_colors_and_lines(colorPreset)
        elif colorPreset != None:
            colors = colorPreset
        else:
            cmap = matplotlib.cm.get_cmap('jet')
            for i in range(self.main_cfg.KEYPOINTDETECT.NUM_JOINTS):
                colors.append(((np.array(
                        cmap(float(i)/self.main_cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                        255).astype(int)[:3]).tolist())

        preds, maxvals = darkpose.get_final_preds(
                    outputs[1].clamp(0,255).detach().cpu().numpy(), None)

        assert (preds[0].shape[0] <= len(colors)), "colorPreset does not match number of Keypoints!"
        for i,point in enumerate(preds[0]):
            if (maxvals[0][i]) > 50:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, colors[i], thickness=5)
            else:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, (100,100,100), thickness=5)


        return preds, maxvals, img_vis, outputs


    def predictPosesVideos(self, centerDetect, video_path,
            output_dir, frameStart = 0, numberFrames = -1, skeletonPreset = None):

        img_size = self.main_cfg.DATASET.IMAGE_SIZE

        cap = cv2.VideoCapture(video_path)
        cap.set(1,frameStart)
        frameRate = cap.get(cv2.CAP_PROP_FPS)
        os.makedirs(os.path.join(output_dir), exist_ok = True)
        out = cv2.VideoWriter(os.path.join(output_dir,
                    video_path.split('/')[-1].split(".")[0] + ".avi"),
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frameRate,
                    (img_size[0],img_size[1]))

        counter = 0
        with open(os.path.join(output_dir, 'data2D.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

            colors = []
            line_idxs = []
            if isinstance(skeletonPreset, str):
                colors, line_idxs = self.get_colors_and_lines(skeletonPreset)
                self.create_header(writer, skeletonPreset)
            elif skeletonPreset != None:
                colors = skeletonPreset["colors"]
                line_idxs = skeletonPreset["line_idxs"]
            else:
                cmap = matplotlib.cm.get_cmap('jet')
                for i in range(self.main_cfg.KEYPOINTDETECT.NUM_JOINTS):
                    colors.append(((np.array(
                            cmap(float(i)/self.main_cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                            255).astype(int)[:3]).tolist())

            ret = True
            if (numberFrames == -1):
                numberFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while ret and (numberFrames == -1 or counter < numberFrames):
                if counter % 100 == 0:
                    print ("Analysing Frame {}/{}".format(counter, numberFrames))
                counter += 1

                ret, img_orig = cap.read()
                img_downsampled_shape = (self.main_cfg.CENTERDETECT.IMAGE_SIZE,
                                         self.main_cfg.CENTERDETECT.IMAGE_SIZE)
                downsampling_scale = np.array(
                            [float(img_orig.shape[1]/self.main_cfg.CENTERDETECT.IMAGE_SIZE),
                             float(img_orig.shape[0]/self.main_cfg.CENTERDETECT.IMAGE_SIZE)])
                img = ((cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB).astype(np.float32)
                        / 255.0 - self.main_cfg.DATASET.MEAN) / self.main_cfg.DATASET.STD)
                img = cv2.resize(img, img_downsampled_shape)

                img = torch.from_numpy(img.transpose(2,0,1)).cuda().float()
                outputs = centerDetect.model(torch.unsqueeze(img,0))
                center, maxval = darkpose.get_final_preds(
                            outputs[1].clamp(0,255).detach().cpu().numpy(), None)
                center = center.squeeze()
                center =center*(downsampling_scale*2)
                bbox_hw = int(self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
                center[0] = min(max(center[0], bbox_hw), img_size[0]-1-bbox_hw)
                center[1] = min(max(center[1], bbox_hw), img_size[1]-1-bbox_hw)
                center = center.astype(int)

                img = img_orig[center[1]-bbox_hw:center[1]+bbox_hw, center[0]-bbox_hw:center[0]+bbox_hw, :]
                img = ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) /
                            255.0-self.main_cfg.DATASET.MEAN)/self.main_cfg.DATASET.STD)

                img = torch.from_numpy(img)
                img = img.permute(2,0,1).view(1,3,
                            self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                            self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda().float()
                if maxval >= 50:
                    outputs = self.model(img)
                    preds, maxvals = darkpose.get_final_preds(
                                outputs[1].clamp(0,255).detach().cpu().numpy(), None)
                    preds = preds.squeeze()
                    preds = preds*2+center-bbox_hw
                    row = []
                    for point in preds.squeeze():
                        row = row + point.tolist()
                    writer.writerow(row)
                    assert (preds.shape[0] <= len(colors)), "colorPreset does not match number of Keypoints!"
                    for line in line_idxs:
                        self.draw_line(img_orig, line, preds,
                                img_size, colors[line[1]])
                    for j,point in enumerate(preds):
                        if maxvals[0,j] > 50:
                            self.draw_point(img_orig, point, img_size,
                                    colors[j])

                else:
                    row = []
                    for i in range(23*2):
                        row = row + ['NaN']
                    writer.writerow(row)

                out.write(img_orig)

            out.release()
            cap.release()


    def create_header(self, writer, skeletonPreset):
        if skeletonPreset == "Hand":
            joints = ["Pinky_T","Pinky_D","Pinky_M","Pinky_P","Ring_T",
                      "Ring_D","Ring_M","Ring_P","Middle_T","Middle_D",
                      "Middle_M","Middle_P","Index_T","Index_D","Index_M",
                      "Index_P","Thumb_T","Thumb_D","Thumb_M","Thumb_P",
                      "Palm", "Wrist_U","Wrist_R"]
            header = []
            joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in joints))
            header = header + joints
            writer.writerow(header)
        header2 = ['x','y','z']*self.main_cfg.KEYPOINTDETECT.NUM_JOINTS
        writer.writerow(header2)


    def draw_line(self, img, line, points2D, img_size, color):
        array_sum = np.sum(np.array(points2D))
        array_has_nan = np.isnan(array_sum)
        if ((not array_has_nan) and int(points2D[line[0]][0]) < img_size[0]-1
                and int(points2D[line[0]][0]) > 0
                and  int(points2D[line[1]][0]) < img_size[0]-1
                and int(points2D[line[1]][0]) > 0
                and int(points2D[line[0]][1]) < img_size[1]-1
                and int(points2D[line[0]][1]) > 0
                and int(points2D[line[1]][1]) < img_size[1]-1
                and int(points2D[line[1]][1]) > 0):
            cv2.line(img,
                    (int(points2D[line[0]][0]), int(points2D[line[0]][1])),
                    (int(points2D[line[1]][0]), int(points2D[line[1]][1])),
                    color, 1)


    def draw_point(self, img, point, img_size, color):
        array_sum = np.sum(np.array(point))
        array_has_nan = np.isnan(array_sum)
        if ((not array_has_nan) and (point[0] < img_size[0]-1
                and point[0] > 0 and point[1] < img_size[1]-1
                and point[1] > 0)):
            cv2.circle(img, (int(point[0]), int(point[1])),
                    3, color, thickness=3)



    def get_colors_and_lines(self, skeletonPreset):
        colors = []
        line_idxs = []
        if skeletonPreset == "Hand":
            colors = [(255,0,0), (255,0,0),(255,0,0),(255,0,0),
                      (0,255,0),(0,255,0),(0,255,0),(0,255,0),
                      (0,0,255),(0,0,255),(0,0,255),(0,0,255),
                      (255,255,0),(255,255,0),(255,255,0),
                      (255,255,0),(0,255,255),(0,255,255),
                      (0,255,255),(0,255,255),(255,0,255),
                      (100,0,100),(100,0,100)]
            line_idxs = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7],
                         [8,9], [9,10], [10,11], [12,13], [13,14],
                         [14,15], [16,17], [17,18], [18,19],
                         [15,18], [15,19], [15,11], [11,7], [7,3],
                         [3,21], [21,22], [19,22]]
        elif skeletonPreset == "HumanBody":
            colors = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),
                      (100,100,100),(0,255,0),(0,255,0),(0,255,0),
                      (0,255,0),(0,0,255),(0,0,255),(0,0,255),
                      (0,0,255),(100,100,100),(255,0,255),
                      (255,0,255),(255,0,255),(255,0,255),
                      (255,255,0),(255,255,0),(255,255,0),(255,255,0)]
            line_idxs = [[4,5], [5,6], [6,7], [7,8], [4,9], [9,10],
                         [10,11], [11,12], [4,13], [13,14], [14,15],
                         [15,16], [16,17], [13,18], [18,19],
                         [19,20], [20,21]]
        elif skeletonPreset == "RodentBody":
            colors = [(255,0,0), (255,0,0), (255,0,0),
                      (100,100,100), (0,255,0), (0,255,0),
                      (0,255,0), (0,255,0), (0,0,255), (0,0,255),
                      (0,0,255), (0,0,255),(100,100,100),
                      (0,255,255), (0,255,255), (0,255,255),
                      (255,0,255), (255,0,255), (255,0,255),
                      (255,255,0), (255,255,0), (255,255,0)]
            line_idxs = [[0,1], [0,2], [1,2],[1,3], [2,3], [3,7],
                         [7,6], [6,5], [5,4], [3,11], [11,10],
                         [10,9], [9,8], [3,12], [12,15], [15,14],
                         [14,13], [12,18], [18,17], [17,16], [12,19]]

        return colors, line_idxs
