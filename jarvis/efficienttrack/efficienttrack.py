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
import jarvis.efficienttrack.utils as utils
import jarvis.efficienttrack.darkpose as darkpose
from jarvis.logger.logger import NetLogger, AverageMeter

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
                            self.cfg.MAX_LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                            self.cfg.MAX_LEARNING_RATE, momentum=0.9, nesterov=True)



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

    def get_lates_weights(self):
        model_dir = ''
        if self.mode == 'CenterDetect' or self.mode == 'CenterDetectInference':
            model_dir = 'CenterDetect'
        else:
            model_dir = 'KeypointDetect'
        search_path = os.path.join(self.main_cfg.PROJECTS_ROOT_PATH,
                                   self.main_cfg.PROJECT_NAME, 'models', model_dir)
        dirs = os.listdir(search_path)
        dirs = [os.path.join(search_path, d) for d in dirs] # add path to each file
        dirs.sort(key=lambda x: os.path.getmtime(x))
        #dirs = dirs.reverse()
        for weights_dir in dirs:
            weigths_path = os.path.join(weights_dir,
                        f'EfficientTrack-d{self.cfg.COMPOUND_COEF}_final.pth')
            if os.path.isfile(weigths_path):
                return weigths_path
        return None


    def load_latest_weights(self):
        weights_path = self.get_lates_weights()
        if weights_path == None:
            print ("Could not find any weights!")
            return
        self.load_weights(weights_path)


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

        if (self.cfg.USE_ONECYLCLE):
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                        self.cfg.MAX_LEARNING_RATE,
                        steps_per_epoch=len(training_generator),
                        epochs=num_epochs, div_factor=100)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, patience=3, verbose=True,
                        min_lr=0.00005, factor = 0.2)

        for epoch in range(num_epochs):
            progress_bar = tqdm(training_generator)
            for data in progress_bar:
                imgs = data[0].permute(0, 3, 1, 2).float()
                heatmaps = data[1]

                imgs = imgs.cuda()
                heatmaps = list(map(lambda x: x.cuda(non_blocking=True),
                            heatmaps))

                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                heatmaps_losses = self.criterion(outputs, heatmaps)
                loss = 0
                for idx in range(2):
                    if heatmaps_losses[idx] is not None:
                        heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                        loss = loss + heatmaps_loss
                loss.backward()
                self.optimizer.step()
                if self.cfg.USE_ONECYLCLE:
                    self.scheduler.step()

                self.lossMeter.update(loss.item())
                progress_bar.set_description(
                    'Epoch: {}/{}. Loss: {:.5f}'.format(
                        epoch+1, num_epochs, self.lossMeter.read()))


            if not self.cfg.USE_ONECYLCLE:
                self.scheduler.step(self.lossMeter.read())

            self.logger.update_learning_rate(self.optimizer.param_groups[0]['lr'])
            self.logger.update_train_loss(self.lossMeter.read())
            self.lossMeter.reset()

            if (epoch+1) % self.cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                if epoch +1 < num_epochs:
                    self.save_checkpoint(f'EfficientTrack-d{self.cfg.COMPOUND_COEF}_Epoch_{epoch+1}.pth')
                    print('checkpoint...')
                else:
                    self.save_checkpoint(f'EfficientTrack-d{self.cfg.COMPOUND_COEF}_final.pth')

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

                        outputs = self.model(imgs)
                        heatmaps_losses = self.criterion(outputs,
                                    heatmaps)

                        loss = 0
                        for idx in range(2):
                            if heatmaps_losses[idx] is not None:
                                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                                loss = loss + heatmaps_loss

                    outs = outputs[1].clamp(0,255).detach().cpu().numpy()
                    acc = self.calculate_accuracy(outs, keypoints)

                    self.lossMeter.update(loss.item())
                    self.accuracyMeter.update(acc)

                print(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}. Acc: {:1.3f}'.format(
                        epoch+1, num_epochs, self.lossMeter.read(),
                        self.accuracyMeter.read()))

                self.logger.update_val_loss(self.lossMeter.read())
                self.logger.update_val_accuracy(self.accuracyMeter.read())
                self.lossMeter.reset()
                self.accuracyMeter.reset()

                self.model.train()


    def calculate_accuracy(self, outs, gt):
        preds, maxvals = darkpose.get_final_preds(outs,None)
        mask = np.sum(gt,axis = 2)
        masked = np.ma.masked_where(mask == 0,
                    np.linalg.norm((preds+0.5)*2-gt,
                    axis = 2))
        return np.mean(masked)


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
        return preds, maxvals, img_vis


    def predictKeypoints(self, img):
        img_vis = ((img*self.main_cfg.DATASET.STD)+self.main_cfg.DATASET.MEAN)*255
        img= torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.reshape(1,3,self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                              self.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda()
        outputs = self.model(img)
        colors = []
        cmap = matplotlib.cm.get_cmap('jet')
        for i in range(self.main_cfg.KEYPOINTDETECT.NUM_JOINTS):
            colors.append(((np.array(
                    cmap(float(i)/self.main_cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                    255).astype(int)[:3]).tolist())

        preds, maxvals = darkpose.get_final_preds(
                    outputs[1].clamp(0,255).detach().cpu().numpy(), None)

        for i,point in enumerate(preds[0]):
            if (maxvals[0][i]) > 50:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, colors[i], thickness=5)
            else:
                cv2.circle(img_vis, (int(point[0]*2), int(point[1])*2), 2, (100,100,100), thickness=5)

        return preds, maxvals, img_vis
