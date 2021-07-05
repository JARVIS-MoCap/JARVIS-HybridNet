"""
vortex.py
===============
Vortex convenience class, can be used to train and troublshoot the
Vortex module.
"""

import os
import numpy as np
from tqdm.autonotebook import tqdm
import traceback
import cv2
import time

import torch
from torch import nn
import onnx

from .model import VortexBackbone
from .loss import MSELoss
import lib.vortex.utils as utils
from lib.logger.logger import NetLogger, AverageMeter
import lib.vortex.modules.efficienttrack.darkpose as darkpose



class Vortex:
    """
    Vortex convenience class, enables easy training and inference with
    using the Vortex Module.

    :param mode: Select wether the network is loaded in training or inference
                 mode
    :type mode: string
    :param cfg: Handle for the global configuration structure
    :param weights: Path to parameter savefile to be loaded
    :type weights: string, optional
    """
    def __init__(self, mode, cfg, calibPaths,weights = None, efficienttrack_weights = None):
        self.mode = mode
        self.cfg = cfg
        self.model = VortexBackbone(cfg, calibPaths[0], calibPaths[1], efficienttrack_weights)

        if mode  == 'train':
            #Maybe move this to a function in cfg??
            self.logger = NetLogger(os.path.join(self.cfg.logPaths['vortex'], 'Run_Test'))
            self.lossMeter = AverageMeter()
            self.accuracyMeter = AverageMeter()
            self.valLossMeter = AverageMeter()
            self.valAccuracyMeter = AverageMeter()

            self.model_savepath = os.path.join(self.cfg.savePaths['vortex'], 'Run_Test')

            self.load_weights(weights)

            self.criterion = MSELoss()

            if self.cfg.NUM_GPUS > 0:
                self.model = self.model.cuda()
                if self.cfg.NUM_GPUS > 1:
                    self.model = utils.CustomDataParallel(self.model, self.cfg.NUM_GPUS)

            if self.cfg.VORTEX.OPTIMIZER == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), self.cfg.VORTEX.LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), self.cfg.VORTEX.LEARNING_RATE, momentum=0.9, nesterov=True)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True, min_lr=0.00005, factor = 0.2)

        elif mode == 'inference':
            self.load_weights(weights)
            self.model.requires_grad_(False)
            self.model.eval()
            self.model = self.model.cuda()

        elif mode == 'export':
            pass
            #self.model = EfficientTrackBackbone(num_classes=len(self.cfg.DATASET.OBJ_LIST), compound_coef=self.cfg.EFFICIENTTRACK.COMPOUND_COEF, onnx_export = True)
            #self.load_weights(weights)
            #self.model = self.model.cuda()



    def load_weights(self, weights_path = None):
        if weights_path is not None:
            try:
                ret = self.model.load_state_dict(torch.load(weights_path), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')
                print(
                    '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

            print(f'[Info] loaded weights: {os.path.basename(weights_path)}')
        else:
            print('[Info] initializing weights...')
            #utils.init_weights(self.model)


    def train(self, training_generator, val_generator, num_epochs, start_epoch = 0):
        """
        Function to train the network on a given dataset for a set number of epochs.
        Most of the training parameters can be set in the config file.

        :param training_generator: training data generator (default torch data
            generator)
        :type training_generator: TODO
        :param val_generator: validation data generator (default torch data
            generator)
        :type val_generator: TODO
        :param num_epochs: Number of epochs the network is trained for
        :type num_epochs: int
        :param start_epoch: Initial epoch for the training, set this if training
            is continued from an earlier session
        """
        epoch = start_epoch
        best_loss = 1e5
        best_epoch = 0
        self.model.train()
        if (self.cfg.USE_MIXED_PRECISION):
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            progress_bar = tqdm(training_generator)
            for data in progress_bar:
                imgs = data[0].permute(0,1,4,2,3).float()
                keypoints = data[1]
                centerHM = data[2]
                center3D = data[3]
                heatmap3D = data[4]

                if self.cfg.NUM_GPUS == 1:
                    imgs = imgs.cuda()
                    keypoints = keypoints.cuda()
                    centerHM = centerHM.cuda()
                    center3D = center3D.cuda()
                    heatmap3D = heatmap3D.cuda()

                self.optimizer.zero_grad()
                if (self.cfg.USE_MIXED_PRECISION):
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs, centerHM, center3D)
                        loss = self.criterion(outputs[0], heatmap3D)
                        loss = loss.mean()
                        acc = torch.mean(torch.sqrt(torch.sum((keypoints-outputs[2])**2, dim = 2)))
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                else:
                    outputs = self.model(imgs, centerHM, center3D)
                    loss = self.criterion(outputs[0], heatmap3D)
                    loss = loss.mean()
                    acc = torch.mean(torch.sqrt(torch.sum((keypoints-outputs[2])**2, dim = 2)))

                    loss.backward()
                    self.optimizer.step()

                self.lossMeter.update(loss.item())
                self.accuracyMeter.update(acc.item())

                progress_bar.set_description(
                    'Epoch: {}/{}. Loss: {:.4f}. Acc: {:.2f}'.format(
                        epoch, num_epochs, self.lossMeter.read(), self.accuracyMeter.read()))


            self.logger.update_train_loss(self.lossMeter.read())
            self.scheduler.step(self.lossMeter.read())

            self.lossMeter.reset()
            self.accuracyMeter.reset()

            if epoch % self.cfg.VORTEX.CHECKPOINT_SAVE_INTERVAL == 0 and epoch > 0:
                self.save_checkpoint(f'Vortex-d_{epoch}.pth')
                print('checkpoint...')

            if epoch % self.cfg.VORTEX.VAL_INTERVAL == 0:
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

                        if self.cfg.NUM_GPUS == 1:
                            imgs = imgs.cuda()
                            keypoints = keypoints.cuda()
                            centerHM = centerHM.cuda()
                            center3D = center3D.cuda()
                            heatmap3D = heatmap3D.cuda()

                        if (self.cfg.USE_MIXED_PRECISION):
                            with torch.cuda.amp.autocast():
                                outputs = self.model(imgs, centerHM, center3D)
                                loss = self.criterion(outputs[0], heatmap3D)
                                loss = loss.mean()
                                acc = torch.mean(torch.sqrt(torch.sum((keypoints-outputs[2])**2, dim = 2)))
                        else:
                            outputs = self.model(imgs, centerHM, center3D)
                            loss = self.criterion(outputs[0], heatmap3D)
                            loss = loss.mean()
                            acc = torch.mean(torch.sqrt(torch.sum((keypoints-outputs[2])**2, dim = 2)))

                        self.valLossMeter.update(loss.item())
                        self.valAccuracyMeter.update(acc.item())

            print(
                'Val. Epoch: {}/{}. Loss: {:.3f}. Acc: {:.2f}'.format(
                    epoch, num_epochs, self.valLossMeter.read(),  self.valAccuracyMeter.read()))


            self.logger.update_val_loss(avg_val_loss/len(val_generator))
            self.valLossMeter.reset()
            self.valAccuracyMeter.reset()

            if loss + self.cfg.VORTEX.EARLY_STOPPING_MIN_DELTA < best_loss  and self.cfg.VORTEX.USE_EARLY_STOPPING:
                best_loss = loss
                best_epoch = epoch
                #self.save_checkpoint(f'Vortex_{epoch}.pth')

            self.model.train()

            # Early stopping
            if epoch - best_epoch > self.cfg.VORTEX.EARLY_STOPPING_PATIENCE > 0 and self.cfg.VORTEX.USE_EARLY_STOPPING:
                print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                break


    def save_checkpoint(self, name):
        if isinstance(self.model, utils.CustomDataParallel):
            torch.save(self.module.model.state_dict(), os.path.join(self.model_savepath, name))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.model_savepath, name))


    def export_to_onnx(self, savefile_name, input_size = 256, export_params = True, opset_version = 9):
        input = torch.zeros(1,3,input_size,input_size).cuda()
        torch.onnx.export(self.model, input, savefile_name, input_names=['input'],
        	              output_names=['output'], export_params=True, opset_version=9, do_constant_folding = True)
        onnx_model = onnx.load(savefile_name)
        onnx.checker.check_model(onnx_model)
