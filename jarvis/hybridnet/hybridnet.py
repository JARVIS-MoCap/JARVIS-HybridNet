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
import cv2
import matplotlib
import streamlit as st

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import HybridNetBackbone
from .loss import MSELoss
import jarvis.utils.utils as utils
from jarvis.logger.logger import NetLogger, AverageMeter
import jarvis.efficienttrack.darkpose as darkpose

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

            self.load_weights(weights)

            self.criterion = MSELoss()
            self.model = self.model.cuda()

            if self.cfg.HYBRIDNET.OPTIMIZER == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(),
                            self.cfg.HYBRIDNET.MAX_LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                            self.cfg.HYBRIDNET.MAX_LEARNING_RATE,
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
        if weights_path == 'latest':
            weights_path =  self.get_latest_weights()
        if weights_path is not None:
            if os.path.isfile(weights_path):
                state_dict = torch.load(weights_path)
                self.model.load_state_dict(state_dict, strict=True)
                print(f'[Info] loaded weights: {weights_path}')
                return True
            else:
                return False

        else:
            return True

    def get_latest_weights(self):
        search_path = os.path.join(self.cfg.PARENT_DIR, 'projects',
                                   self.cfg.PROJECT_NAME, 'models', 'HybridNet')
        dirs = os.listdir(search_path)
        dirs = [os.path.join(search_path, d) for d in dirs] # add path to each file
        dirs.sort(key=lambda x: os.path.getmtime(x))
        dirs.reverse()
        for weights_dir in dirs:
            weigths_path = os.path.join(weights_dir,
                        f'HybridNet-d{self.cfg.KEYPOINTDETECT.COMPOUND_COEF}_final.pth')
            if os.path.isfile(weigths_path):
                return weigths_path
        return None


    def train(self, training_set, validation_set, num_epochs, start_epoch = 0, streamlitWidgets = None):
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
        self.model.train()

        latest_train_loss = 0
        latest_train_acc = 0
        latest_val_loss = 0
        latest_val_acc = 0

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        if streamlitWidgets != None:
            streamlitWidgets[2].markdown(f"Epoch {1}/{num_epochs}")

        if (self.cfg.HYBRIDNET.USE_ONECYLCLE):
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                        self.cfg.HYBRIDNET.MAX_LEARNING_RATE,
                        steps_per_epoch=len(training_generator),
                        epochs=num_epochs, div_factor=100)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, patience=3, verbose=True,
                        min_lr=0.00005, factor = 0.2)

        for epoch in range(num_epochs):
            progress_bar = tqdm(training_generator)
            for counter, data in enumerate(progress_bar):
                imgs = data[0].permute(0,1,4,2,3).float()
                keypoints = data[1]
                centerHM = data[2]
                center3D = data[3]
                heatmap3D = data[4]
                cameraMatrices = data[5]
                intrinsicMatrices = data[6]
                distortionCoefficients = data[7]

                imgs = imgs.cuda()
                keypoints = keypoints.cuda()
                centerHM = centerHM.cuda()
                center3D = center3D.cuda()
                heatmap3D = heatmap3D.cuda()
                cameraMatrices = cameraMatrices.cuda()
                intrinsicMatrices = intrinsicMatrices.cuda()
                distortionCoefficients = distortionCoefficients.cuda()
                img_size = torch.tensor(self.cfg.DATASET.IMAGE_SIZE).cuda()


                self.optimizer.zero_grad()
                outputs = self.model(imgs,
                                     img_size,
                                     centerHM,
                                     center3D,
                                     cameraMatrices,
                                     intrinsicMatrices,
                                     distortionCoefficients)
                loss = self.criterion(outputs[0], heatmap3D)
                loss = loss.mean()

                acc = 0
                count = 0
                for i,keypoints_batch in enumerate(keypoints):
                    for j,keypoint in enumerate(keypoints_batch):
                        if keypoint[0] != 0 or keypoint[1] != 0  or keypoint[2] != 0:
                            acc += torch.sqrt(torch.sum(
                                        (keypoint-outputs[2][i][j])**2))
                            count += 1
                acc = acc/count

                loss.backward()
                self.optimizer.step()
                if self.cfg.HYBRIDNET.USE_ONECYLCLE:
                    self.scheduler.step()

                self.lossMeter.update(loss.item())
                self.accuracyMeter.update(acc.item())

                progress_bar.set_description(
                    'Epoch: {}/{}. Loss: {:.4f}. Acc: {:.2f}'.format(
                        epoch, num_epochs, self.lossMeter.read(),
                        self.accuracyMeter.read()))
                if streamlitWidgets != None:
                    streamlitWidgets[1].progress(float(counter+1)/float(len(training_generator)))

            latest_train_loss = self.lossMeter.read()
            train_losses.append(latest_train_loss)
            latest_train_acc = self.accuracyMeter.read()
            train_accs.append(latest_train_acc)
            self.logger.update_train_loss(self.lossMeter.read())
            self.logger.update_train_accuracy(self.accuracyMeter.read())

            if not self.cfg.HYBRIDNET.USE_ONECYLCLE:
                self.scheduler.step(self.lossMeter.read())

            self.lossMeter.reset()
            self.accuracyMeter.reset()

            if (epoch + 1) % self.cfg.HYBRIDNET.CHECKPOINT_SAVE_INTERVAL == 0:
                if epoch + 1 < num_epochs:
                    self.save_checkpoint(f'HybridNet-d{self.cfg.KEYPOINTDETECT.COMPOUND_COEF}_Epoch_{epoch+1}.pth')
                    print('checkpoint...')
            if epoch + 1 == num_epochs:
                self.save_checkpoint(f'HybridNet-d{self.cfg.KEYPOINTDETECT.COMPOUND_COEF}_final.pth')

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
                        intrinsicMatrices = data[6]
                        distortionCoefficients = data[7]

                        imgs = imgs.cuda()
                        keypoints = keypoints.cuda()
                        centerHM = centerHM.cuda()
                        center3D = center3D.cuda()
                        heatmap3D = heatmap3D.cuda()
                        cameraMatrices = cameraMatrices.cuda()
                        intrinsicMatrices = intrinsicMatrices.cuda()
                        distortionCoefficients = distortionCoefficients.cuda()
                        img_size = torch.tensor(self.cfg.DATASET.IMAGE_SIZE).cuda()

                        outputs = self.model(imgs,
                                             img_size,
                                             centerHM,
                                             center3D,
                                             cameraMatrices,
                                             intrinsicMatrices,
                                             distortionCoefficients)
                        loss = self.criterion(outputs[0], heatmap3D)
                        loss = loss.mean()
                        acc = 0
                        count = 0
                        for i,keypoints_batch in enumerate(keypoints):
                            for j,keypoint in enumerate(keypoints_batch):
                                if keypoint[0] != 0 or keypoint[1] != 0  or keypoint[2] != 0:
                                    acc += torch.sqrt(torch.sum(
                                                (keypoint-outputs[2][i][j])**2))
                                    count += 1
                        acc = acc/count

                        self.lossMeter.update(loss.item())
                        self.accuracyMeter.update(acc.item())

            print(
                'Val. Epoch: {}/{}. Loss: {:.3f}. Acc: {:.2f}'.format(
                    epoch+1, num_epochs, self.lossMeter.read(),
                    self.accuracyMeter.read()))

            latest_val_loss = self.lossMeter.read()
            val_losses.append(latest_val_loss)
            latest_val_acc = self.accuracyMeter.read()
            val_accs.append(latest_val_acc)
            self.logger.update_val_loss(self.lossMeter.read())
            self.logger.update_val_accuracy(self.accuracyMeter.read())
            self.lossMeter.reset()
            self.accuracyMeter.reset()
            self.model.train()
            if streamlitWidgets != None:
                streamlitWidgets[0].progress(float(epoch+1)/float(num_epochs))
                streamlitWidgets[2].markdown(f"Epoch {epoch+1}/{num_epochs}")
                streamlitWidgets[3].line_chart({'Train Loss': train_losses, 'Val Loss': val_losses})
                streamlitWidgets[4].line_chart({'Train Accuracy [mm]': train_accs, 'Val Accuracy [mm]': val_accs})
                st.session_state['HybridNet/' + self.training_mode + '/Train Loss'] = train_losses
                st.session_state['HybridNet/' + self.training_mode + '/Val Loss'] = val_losses
                st.session_state['HybridNet/' + self.training_mode + '/Train Accuracy'] = train_accs
                st.session_state['HybridNet/' + self.training_mode + '/Val Accuracy'] = val_accs
                st.session_state['results_available'] = True

        final_results = {'train_loss': latest_train_loss,
                         'train_acc': latest_train_acc,
                         'val_loss': latest_val_loss,
                         'val_acc': latest_val_acc}
        return final_results


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
        self.training_mode = mode
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
