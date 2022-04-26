"""
efficienttrack.py
=================
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
import pandas as pd
import streamlit as st

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import EfficientTrackBackbone
from .loss import HeatmapLoss
import jarvis.efficienttrack.utils as utils
import jarvis.efficienttrack.darkpose as darkpose
from jarvis.utils.logger import NetLogger, AverageMeter

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
                    model_size=self.cfg.MODEL_SIZE,
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

            self.found_weights = self.load_weights(weights)

            self.criterion = HeatmapLoss(self.cfg, self.mode)
            self.model = self.model.cuda()

            if self.cfg.OPTIMIZER == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(),
                            self.cfg.MAX_LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                            self.cfg.MAX_LEARNING_RATE, momentum=0.9,
                            nesterov=True)

        elif mode == 'KeypointDetectInference' or 'CenterDetectInference':
            self.load_weights(weights)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                print ("[Info] No GPU available, model is compiled on CPU.")
            self.model.requires_grad_(False)
            self.model.eval()


    def load_weights(self, weights_path = None):
        if weights_path == 'latest':
            weights_path =  self.get_latest_weights()
        if weights_path is not None:
            if os.path.isfile(weights_path):
                if torch.cuda.is_available():
                    pretrained_dict = torch.load(weights_path)
                else:
                    pretrained_dict = torch.load(weights_path,
                                map_location=torch.device('cpu'))
                if (self.mode == "KeypointDetect" and
                            pretrained_dict['final_conv1.weight'].shape[0]
                            != self.cfg.NUM_JOINTS):
                    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                if not k in ['final_conv1.weight',
                                'final_conv2.weight']}
                self.model.load_state_dict(pretrained_dict, strict=False)
                print(f'Successfully loaded weights: {weights_path}')
                return True
            else:
                return False
        else:
            utils.init_weights(self.model)
            return True


    def load_ecoset_pretrain(self):
        weights_path = os.path.join(self.main_cfg.PARENT_DIR, 'pretrained',
                    'EcoSet', f'EfficientTrack-{self.cfg.MODEL_SIZE}.pth')
        if os.path.isfile(weights_path):
            if torch.cuda.is_available():
                pretrained_dict = torch.load(weights_path)
            else:
                pretrained_dict = torch.load(weights_path,
                            map_location=torch.device('cpu'))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if not k in ['final_conv1.weight', 'final_conv2.weight',
                        'first_conv.pointwise_conv.bias',
                        'first_conv.gn.weight', 'first_conv.gn.bias',
                        'first_conv.pointwise_conv.weight']}
            self.model.load_state_dict(pretrained_dict, strict=False)
            print(f'Successfully loaded EcoSet weights: {weights_path}')
            return True
        else:
            print(f'Could not load EcoSet weights: {weights_path}')
            return False


    def load_pose_pretrain(self, pose):
        if self.mode == 'CenterDetect' or self.mode == 'CenterDetectInference':
            weights_name = f"EfficientTrack_Center-{self.cfg.MODEL_SIZE}.pth"
        else:
            weights_name = f"EfficientTrack_Keypoints-{self.cfg.MODEL_SIZE}.pth"
        weights_path = os.path.join(self.main_cfg.PARENT_DIR, 'pretrained',
                    pose, weights_name)
        if os.path.isfile(weights_path):
            if torch.cuda.is_available():
                pretrained_dict = torch.load(weights_path)
            else:
                pretrained_dict = torch.load(weights_path,
                            map_location=torch.device('cpu'))
            if (self.mode == "KeypointDetect"
                        and pretrained_dict['final_conv1.weight'].shape[0]
                        != self.cfg.NUM_JOINTS):
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if not k in ['final_conv1.weight',
                            'final_conv2.weight']}
            self.model.load_state_dict(pretrained_dict, strict=False)
            print(f'Successfully loaded {pose} weights: {weights_path}')
            return True
        else:
            print(f'Could not load {pose} weights: {weights_path}')
            return False


    def get_latest_weights(self):
        model_dir = ''
        if self.mode == 'CenterDetect' or self.mode == 'CenterDetectInference':
            model_dir = 'CenterDetect'
        else:
            model_dir = 'KeypointDetect'
        search_path = os.path.join(self.main_cfg.PARENT_DIR, 'projects',
                                   self.main_cfg.PROJECT_NAME, 'models',
                                   model_dir)
        dirs = os.listdir(search_path)
        dirs = [os.path.join(search_path, d) for d in dirs]
        dirs.sort(key=lambda x: os.path.getmtime(x))
        dirs.reverse()
        for weights_dir in dirs:
            weigths_path = os.path.join(weights_dir,
                        f'EfficientTrack-{self.cfg.MODEL_SIZE}_final.pth')
            if os.path.isfile(weigths_path):
                return weigths_path
        return None


    def load_latest_weights(self):
        weights_path = self.get_lates_weights()
        if weights_path == None:
            print ("Could not find any weights!")
            return
        self.load_weights(weights_path)


    def train(self, training_set, validation_set, num_epochs, start_epoch = 0,
                streamlitWidgets = None):
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

        epoch = start_epoch #TODO: actually use this and make it work properly with onecylce
        self.model.train()

        latest_train_loss = 0
        latest_val_loss = 0
        latest_val_acc = 0

        train_losses = []
        val_losses = []
        val_accs = []

        if (self.cfg.USE_ONECYLCLE):
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                        self.cfg.MAX_LEARNING_RATE,
                        steps_per_epoch=len(training_generator),
                        epochs=num_epochs, div_factor=100)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, patience=3, verbose=True,
                        min_lr=0.00005, factor = 0.2)

        if streamlitWidgets != None:
            streamlitWidgets[2].markdown(f"Epoch {1}/{num_epochs}")

        for epoch in range(num_epochs):
            progress_bar = tqdm(training_generator)
            for count,data in enumerate(progress_bar):
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
                if streamlitWidgets != None:
                    streamlitWidgets[1].progress(float(count + 1)
                                / float(len(training_generator)))

            if not self.cfg.USE_ONECYLCLE:
                self.scheduler.step(self.lossMeter.read())

            self.logger.update_learning_rate(
                        self.optimizer.param_groups[0]['lr'])
            self.logger.update_train_loss(self.lossMeter.read())
            latest_train_loss = self.lossMeter.read()
            train_losses.append(latest_train_loss)
            self.lossMeter.reset()

            if (epoch + 1) % self.cfg.CHECKPOINT_SAVE_INTERVAL == 0:
                if epoch + 1 < num_epochs:
                    self.save_checkpoint(f'EfficientTrack-'
                                f'{self.cfg.MODEL_SIZE}_Epoch_{epoch+1}.pth')
                    print('checkpoint...')
            if epoch + 1 == num_epochs:
                self.save_checkpoint(f'EfficientTrack-'
                            f'{self.cfg.MODEL_SIZE}_final.pth')

            if (epoch + 1) % self.cfg.VAL_INTERVAL == 0:
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
                    if (acc != -1):
                        self.accuracyMeter.update(acc)

                print(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}. Acc: {:1.3f}'.format(
                        epoch+1, num_epochs, self.lossMeter.read(),
                        self.accuracyMeter.read()))

                latest_val_loss = self.lossMeter.read()
                val_losses.append(latest_val_loss)
                latest_val_acc = self.accuracyMeter.read()
                if np.isnan(latest_val_acc):
                    latest_val_acc = 0
                val_accs.append(latest_val_acc)
                self.logger.update_val_loss(self.lossMeter.read())
                self.logger.update_val_accuracy(self.accuracyMeter.read())
                self.lossMeter.reset()
                self.accuracyMeter.reset()

                self.model.train()
                if streamlitWidgets != None:
                    streamlitWidgets[0].progress(float(epoch + 1)
                                / float(num_epochs))
                    streamlitWidgets[2].markdown(f"Epoch {epoch+1}/{num_epochs}")
                    streamlitWidgets[3].line_chart({'Train Loss': train_losses,
                                'Val Loss': val_losses})
                    streamlitWidgets[4].line_chart(
                                {'Val Accuracy [px]': val_accs})
                    st.session_state[self.mode+'/'+'Train Loss'] = train_losses
                    st.session_state[self.mode+'/'+'Val Loss'] = val_losses
                    st.session_state[self.mode+'/'+'Val Accuracy'] = val_accs
                    st.session_state['results_available'] = True


        final_results = {'train_loss': latest_train_loss,
                         'val_loss': latest_val_loss,
                         'val_acc': latest_val_acc}
        return final_results


    def calculate_accuracy(self, outs, gt):
        preds, maxvals = darkpose.get_final_preds(outs,None)
        mask = np.sum(gt,axis = 2)
        masked = np.ma.masked_where(mask == 0,
                    np.linalg.norm((preds+0.5)*2-gt,
                    axis = 2))
        if (masked.mask.all()):
            return -1
        return np.nanmean(masked)


    def save_checkpoint(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.model_savepath, name))
