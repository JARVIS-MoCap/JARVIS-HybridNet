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
            self.model.load_state_dict(torch.load(weights_path), strict=True)
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
        #try:
        for epoch in range(num_epochs):
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
        preds, maxvals = darkpose.get_final_preds(outputs[1].clamp(0,255).detach().cpu().numpy(), None)
        for i,point in enumerate(preds[0]):
            if (maxvals[0][i]) > 10:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, (255,0,0), thickness=5)
            else:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, (100,0,0), thickness=5)

        heatmap = cv2.resize(outputs[1].clamp(0,255).detach().cpu().numpy()[0][0]/255., (img_shape[1],img_shape[0]), interpolation=cv2.cv2.INTER_NEAREST)
        return preds, maxvals, img_vis, heatmap


    def predictKeypoints(self, img):
        img_vis = ((img*self.main_cfg.DATASET.STD)+self.main_cfg.DATASET.MEAN)*255
        img= torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.reshape(1,3,self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda()
        outputs = self.model(img)
        colors = [(255,0,0), (255,0,0),(255,0,0),(255,0,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(255,255,0),(255,255,0),(255,255,0), (255,255,0),
                  (0,255,255),(0,255,255),(0,255,255),(0,255,255), (255,0,255),(100,0,100),(100,0,100)]
        preds, maxvals = darkpose.get_final_preds(outputs[1].clamp(0,255).detach().cpu().numpy(), None)
        for i,point in enumerate(preds[0]):
            if (maxvals[0][i]) > 10:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, colors[i], thickness=5)
            else:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, (100,0,0), thickness=5)

        return preds, maxvals, img_vis
