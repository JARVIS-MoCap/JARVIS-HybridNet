"""
hybridnet.py
===============
HybridNet convenience class, can be used to train and troublshoot the
HybridNet module.
"""

import os
import numpy as np
from tqdm.autonotebook import tqdm
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import HybridNetBackbone
from .loss import MSELoss
import lib.utils.utils as utils
from lib.logger.logger import NetLogger, AverageMeter
import lib.hybridnet.efficienttrack.darkpose as darkpose

import warnings
#Filter out weird pytorch floordiv deprecation warning, don't know where it's
#coming from so can't really fix it
warnings.filterwarnings("ignore", category=UserWarning)

class HybridNet:
    """
    hybridNet convenience class, enables easy training and inference with
    using the HybridNet Module.

    :param mode: Select wether the network is loaded in training or inference
                 mode
    :type mode: string
    :param cfg: Handle for the global configuration structure
    :param weights: Path to parameter savefile to be loaded
    :type weights: string, optional
    """
    def __init__(self, mode, cfg, weights = None, efficienttrack_weights = None,
                 run_name = None):
        self.mode = mode
        self.cfg = cfg
        self.model = HybridNetBackbone(cfg, efficienttrack_weights)

        if mode  == 'train':
            if run_name == None:
                run_name = "Run_" + time.strftime("%Y%m%d-%H%M%S")

            self.model_savepath = os.path.join(self.cfg.savePaths['HybridNet'],
                        run_name)
            os.makedirs(self.model_savepath, exist_ok=True)

            self.logger = NetLogger(os.path.join(self.cfg.logPaths['HybridNet'],
                        run_name))
            self.lossMeter = AverageMeter()
            self.accuracyMeter = AverageMeter()
            self.valLossMeter = AverageMeter()
            self.valAccuracyMeter = AverageMeter()

            self.load_weights(weights)

            self.criterion = MSELoss()
            self.model = self.model.cuda()

            if self.cfg.HYBRIDNET.OPTIMIZER == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(),
                            self.cfg.HYBRIDNET.LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                            self.cfg.HYBRIDNET.LEARNING_RATE,
                            momentum=0.9, nesterov=True)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, patience=3, verbose=True,
                        min_lr=0.00005, factor = 0.2)
            self.set_training_mode('all')

        elif mode == 'inference':
            self.load_weights(weights)
            self.model.requires_grad_(False)
            self.model.eval()
            self.model = self.model.cuda()


    def load_weights(self, weights_path = None):
        if weights_path is not None:
            state_dict = torch.load(weights_path)
            self.model.load_state_dict(state_dict, strict=False)
            print(f'[Info] loaded weights: {os.path.basename(weights_path)}')
        else:
            print('[Info] initializing weights...')
            #utils.init_weights(self.model)


    def train(self, training_set, validation_set, num_epochs, start_epoch = 0):
        """
        Function to train the network on a given dataset for a set number of
        epochs. Most of the training parameters can be set in the config file.

        :param training_generator: training data generator (default torch data
                                   generator)
        :type training_generator: TODO
        :param val_generator: validation data generator (default torch data
                              generator)
        :type val_generator: TODO
        :param num_epochs: Number of epochs the network is trained for
        :type num_epochs: int
        :param start_epoch: Initial epoch for the training, set this if
                            training is continued from an earlier session
        """
        training_generator = DataLoader(
                    training_set,
                    batch_size = self.cfg.HYBRIDNET.BATCH_SIZE,
                    shuffle = True,
                    num_workers =  self.cfg.DATALOADER_NUM_WORKERS,
                    pin_memory = True)

        val_generator = DataLoader(
                    validation_set,
                    batch_size = self.cfg.HYBRIDNET.BATCH_SIZE,
                    shuffle = False,
                    num_workers =  self.cfg.DATALOADER_NUM_WORKERS,
                    pin_memory = True)
        epoch = start_epoch
        best_loss = 1e5
        best_epoch = 0
        self.model.train()
        if self.cfg.USE_MIXED_PRECISION:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            progress_bar = tqdm(training_generator)
            for data in progress_bar:
                imgs = data[0].permute(0,1,4,2,3).float()
                keypoints = data[1]
                centerHM = data[2]
                center3D = data[3]
                heatmap3D = data[4]
                cameraMatrices = data[5]

                imgs = imgs.cuda()
                keypoints = keypoints.cuda()
                centerHM = centerHM.cuda()
                center3D = center3D.cuda()
                heatmap3D = heatmap3D.cuda()
                cameraMatrices = cameraMatrices.cuda()

                self.optimizer.zero_grad()
                if self.cfg.USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs, centerHM, center3D,
                                    cameraMatrices)
                        loss = self.criterion(outputs[0], heatmap3D)
                        loss = loss.mean()
                        acc = torch.mean(torch.sqrt(torch.sum(
                                    (keypoints-outputs[2])**2, dim = 2)))
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                else:
                    outputs = self.model(imgs, centerHM, center3D,
                                         cameraMatrices)
                    loss = self.criterion(outputs[0], heatmap3D)
                    loss = loss.mean()
                    acc = torch.mean(torch.sqrt(torch.sum(
                                (keypoints-outputs[2])**2, dim = 2)))

                    loss.backward()
                    self.optimizer.step()

                self.lossMeter.update(loss.item())
                self.accuracyMeter.update(acc.item())

                progress_bar.set_description(
                    'Epoch: {}/{}. Loss: {:.4f}. Acc: {:.2f}'.format(
                        epoch, num_epochs, self.lossMeter.read(),
                        self.accuracyMeter.read()))


            self.logger.update_train_loss(self.lossMeter.read())
            self.logger.update_train_accuracy(self.accuracyMeter.read())
            self.scheduler.step(self.lossMeter.read())

            self.lossMeter.reset()
            self.accuracyMeter.reset()

            if (epoch % self.cfg.HYBRIDNET.CHECKPOINT_SAVE_INTERVAL == 0
                        and epoch > 0):
                self.save_checkpoint(f'HybridNet-d_{epoch}.pth')
                print('checkpoint...')

            if epoch % self.cfg.HYBRIDNET.VAL_INTERVAL == 0:
                self.model.eval()
                avg_val_loss = 0
                avg_val_acc = 0
                for data in val_generator:
                    with torch.no_grad():
                        imgs = data[0].permute(0,1,4,2,3).float()
                        keypoints = data[1]
                        centerHM = data[2]
                        center3D = data[3]
                        heatmap3D = data[4]
                        cameraMatrices = data[5]

                        imgs = imgs.cuda()
                        keypoints = keypoints.cuda()
                        centerHM = centerHM.cuda()
                        center3D = center3D.cuda()
                        heatmap3D = heatmap3D.cuda()
                        cameraMatrices = cameraMatrices.cuda()


                        if self.cfg.USE_MIXED_PRECISION:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(imgs, centerHM, center3D,
                                                     cameraMatrices)
                                loss = self.criterion(outputs[0], heatmap3D)
                                loss = loss.mean()
                                acc = torch.mean(torch.sqrt(torch.sum(
                                        (keypoints-outputs[2])**2, dim = 2)))
                        else:
                            outputs = self.model(imgs, centerHM, center3D,
                                                 cameraMatrices)
                            loss = self.criterion(outputs[0], heatmap3D)
                            loss = loss.mean()
                            acc = torch.mean(torch.sqrt(torch.sum(
                                        (keypoints-outputs[2])**2, dim = 2)))

                        self.valLossMeter.update(loss.item())
                        self.valAccuracyMeter.update(acc.item())

            print(
                'Val. Epoch: {}/{}. Loss: {:.3f}. Acc: {:.2f}'.format(
                    epoch, num_epochs, self.valLossMeter.read(),
                    self.valAccuracyMeter.read()))


            self.logger.update_val_loss(self.valLossMeter.read())
            self.logger.update_val_accuracy(self.valAccuracyMeter.read())
            self.valLossMeter.reset()
            self.valAccuracyMeter.reset()

            if (loss + self.cfg.HYBRIDNET.EARLY_STOPPING_MIN_DELTA < best_loss
                        and self.cfg.HYBRIDNET.USE_EARLY_STOPPING):
                best_loss = loss
                best_epoch = epoch
                #self.save_checkpoint(f'Vortex_{epoch}.pth')

            self.model.train()

            # Early stopping
            if (epoch-best_epoch > self.cfg.HYBRIDNET.EARLY_STOPPING_PATIENCE > 0
                        and self.cfg.HYBRIDNET.USE_EARLY_STOPPING):
                print('[Info] Stop training at epoch {}. The lowest loss '
                      'achieved is {}'.format(epoch, best_loss))
                break


    def save_checkpoint(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.model_savepath, name))

    def set_training_mode(self, mode):
        """
        Selects which parts of the network will be trained.
        :param mode: 'all': The whole network will be trained
                     'bifpn': The whole network except the efficientnet backbone
                              will be trained
                     'last_layers': The 3D network and the output layers of the
                                    2D network will be trained
                     '3D_only': Only the 3D network will be trained
        """
        if mode == 'all':
            self.model.effTrack.requires_grad_(True)
        elif mode == 'bifpn':
            self.model.effTrack.requires_grad_(True)
            self.model.effTrack.backbone_net.requires_grad_(False)
        elif mode == 'last_layers':
            self.model.effTrack.requires_grad_(True)
            self.model.effTrack.bifpn.requires_grad_(False)
            self.model.effTrack.backbone_net.requires_grad_(False)
        elif mode == '3D_only':
            self.model.effTrack.requires_grad_(False)
