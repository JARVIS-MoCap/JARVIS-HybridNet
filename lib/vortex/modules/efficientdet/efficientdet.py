"""
efficientdet.py
===============
EfficientDet convenience class, can be used to train and troublshoot the
EfficientDet module.
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
import onnx

from .model import EfficientDetBackbone
from .loss import FocalLoss
import lib.vortex.modules.efficientdet.utils as utils
from lib.logger.logger import NetLogger, AverageMeter



class EfficientDet:
    """
    EfficientDet convenience class, enables easy training and inference with
    using the EfficientDet Module.

    :param mode: Select wether the network is loaded in training or inference
                 mode
    :type mode: string
    :param cfg: Handle for the global configuration structure
    :param weights: Path to parameter savefile to be loaded
    :type weights: string, optional
    """
    def __init__(self, mode, cfg, weights = None, run_name = None):
        self.mode = mode
        self.cfg = cfg
        self.model = EfficientDetBackbone(num_classes=len(self.cfg.DATASET.OBJ_LIST), compound_coef=self.cfg.EFFICIENTDET.COMPOUND_COEF,
        ratios=eval(self.cfg.EFFICIENTDET.ANCHOR_RATIOS), scales=eval(self.cfg.EFFICIENTDET.ANCHOR_SCALES))

        if mode  == 'train':
            if run_name == None:
                run_name = "Run_" + time.strftime("%Y%m%d-%H%M%S")

            self.model_savepath = os.path.join(self.cfg.savePaths['efficientdet'], run_name)
            os.makedirs(self.model_savepath, exist_ok=True)

            self.logger = NetLogger(os.path.join(self.cfg.logPaths['efficientdet'], run_name), ['Class Loss', 'Reg Loss', 'Total Loss'])
            self.classLossMeter = AverageMeter()
            self.regLossMeter = AverageMeter()
            self.totalLossMeter = AverageMeter()

            self.load_weights(weights)

            self.criterion = FocalLoss()

            if self.cfg.NUM_GPUS > 0:
                self.model = self.model.cuda()
                if self.cfg.NUM_GPUS > 1:
                    self.model = utils.CustomDataParallel(self.model, self.cfg.NUM_GPUS)

            if self.cfg.EFFICIENTDET.OPTIMIZER == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), self.cfg.EFFICIENTDET.LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), self.cfg.EFFICIENTDET.LEARNING_RATE, momentum=0.9, nesterov=True)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True, min_lr=0.00005, factor = 0.2)

        elif mode == 'inference':
            self.model.requires_grad_(False)
            self.model.load_state_dict(torch.load(weights))
            self.model.requires_grad_(False)
            self.model.eval()
            self.model = self.model.cuda()

        elif mode == 'export':
            self.model = EfficientDetBackbone(num_classes=len(self.cfg.DATASET.OBJ_LIST), compound_coef=self.cfg.EFFICIENTDET.COMPOUND_COEF, onnx_export = True,
            ratios=eval(self.cfg.EFFICIENTDET.ANCHOR_RATIOS), scales=eval(self.cfg.EFFICIENTDET.ANCHOR_SCALES))
            self.load_weights(weights)
            self.model = self.model.cuda()


    def predict_on_image(self, img_path, img = None):
        """
        Do bounding box regression on a single image, or a list of images

        :param img_path: Path of the image to analyse, set to none if already
            loaded image is given instead
        :type img_path: string
        :param img: Image to be analysed (opencv format)
        :type img: np.array, optional
        :return: Dictionary of bbox and class predictions
        """
        #TODO: check if input is list and continue accordingly
        if self.mode == 'train':
            print("Load Model in inference mode!")
            return {"images": [], "predictions": []}
        if img_path != None:
            ori_imgs, framed_imgs, framed_metas = utils.preprocess(img_path, max_size=512) #TODO: make this the resolution from the configuration
        else:
            ori_imgs, framed_imgs, framed_metas = utils.preprocess_img(img, max_size=512)
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)
        with torch.no_grad():
            regression, classification, anchors = self.model(x)

            regressBoxes = utils.BBoxTransform()
            clipBoxes = utils.ClipBoxes()

            threshold = 0.2
            iou_threshold = 0.2
            out = utils.postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)
            out = utils.invert_affine(framed_metas, out)
            out_dict = {"images": ori_imgs, "predictions": out}
            return out_dict


    def visualize_prediction(self,out_dict):
        """
        Takes ouput dictionary from e.g. predict_on_image and plots the bounding
        boxes and class labels.

        :param out_dict: Output dictionary containing bbox and classification results
        :type out_dict: dict
        """
        imgs = out_dict["images"]
        preds = out_dict["predictions"]

        for i in range(len(imgs)):
            for j in range(len(preds[i]['rois'])):
                (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                print (x1, y1, x2, y2)
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                obj = self.cfg.DATASET.OBJ_LIST[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)

            cv2.imshow('frame',imgs[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    def load_weights(self, weights_path = None):
        if weights_path is not None:
            try:
                ret = self.model.load_state_dict(torch.load(weights_path), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')
                print('Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

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
        assert self.cfg.EFFICIENTDET.IMG_SIZE % 128 == 0, "IMG_SIZE must be divisible by 128"

        training_generator = DataLoader(training_set,
                                        batch_size = self.cfg.EFFICIENTDET.BATCH_SIZE,
                                        shuffle = True,
                                        num_workers =  self.cfg.DATALOADER_NUM_WORKERS,
                                        pin_memory = True)

        val_generator = DataLoader(validation_set,
                                        batch_size = self.cfg.EFFICIENTDET.BATCH_SIZE,
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
                imgs = data[0].permute(0, 3, 1, 2).float()
                annot = data[1]
                if self.cfg.NUM_GPUS == 1:
                    imgs = imgs.cuda()
                    annot = annot.cuda()

                self.optimizer.zero_grad()

                if self.cfg.USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        regression, classification, anchors = self.model(imgs)
                        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annot)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()
                        loss = cls_loss + reg_loss
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    regression, classification, anchors = self.model(imgs)
                    cls_loss, reg_loss = self.criterion(classification, regression, anchors, annot)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                    loss.backward()
                    self.optimizer.step()

                self.classLossMeter.update(cls_loss.item())
                self.regLossMeter.update(reg_loss.item())
                self.totalLossMeter.update(loss.item())

                progress_bar.set_description(
                    'Epoch: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                        epoch, num_epochs, self.classLossMeter.read(),
                        self.regLossMeter.read(), self.totalLossMeter.read()))

            self.logger.update_train_loss([self.classLossMeter.read(), self.regLossMeter.read(), self.totalLossMeter.read()])
            self.scheduler.step(self.totalLossMeter.read())

            self.classLossMeter.reset()
            self.regLossMeter.reset()
            self.totalLossMeter.reset()

            if epoch % self.cfg.EFFICIENTDET.CHECKPOINT_SAVE_INTERVAL == 0 and epoch > 0:
                self.save_checkpoint(f'efficientdet-d{self.cfg.EFFICIENTDET.COMPOUND_COEF}_{epoch}.pth')
                print('checkpoint...')

            if epoch % self.cfg.EFFICIENTDET.VAL_INTERVAL == 0:
                self.model.eval()
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data[0].permute(0, 3, 1, 2).float()
                        annot = data[1]

                        if self.cfg.NUM_GPUS == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        if self.cfg.USE_MIXED_PRECISION:
                            with torch.cuda.amp.autocast():
                                regression, classification, anchors = self.model(imgs)
                                cls_loss, reg_loss = self.criterion(classification, regression, anchors, annot)
                                cls_loss = cls_loss.mean()
                                reg_loss = reg_loss.mean()
                                loss = cls_loss + reg_loss
                        else:
                            regression, classification, anchors = self.model(imgs)
                            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annot)
                            cls_loss = cls_loss.mean()
                            reg_loss = reg_loss.mean()
                            loss = cls_loss + reg_loss

                        self.classLossMeter.update(cls_loss.item())
                        self.regLossMeter.update(reg_loss.item())
                        self.totalLossMeter.update(loss.item())

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, num_epochs, self.classLossMeter.read(), self.regLossMeter.read(), self.totalLossMeter.read()))


                if self.totalLossMeter.read() + self.cfg.EFFICIENTDET.EARLY_STOPPING_MIN_DELTA < best_loss:
                    best_loss = self.totalLossMeter.read()
                    best_epoch = epoch
                    self.save_checkpoint(f'efficientdet-d{self.cfg.EFFICIENTDET.COMPOUND_COEF}_{epoch}.pth')

                self.logger.update_val_loss([self.classLossMeter.read(), self.regLossMeter.read(), self.totalLossMeter.read()])
                self.classLossMeter.reset()
                self.regLossMeter.reset()
                self.totalLossMeter.reset()
                self.model.train()

                # Early stopping
                if epoch - best_epoch > self.cfg.EFFICIENTDET.EARLY_STOPPING_PATIENCE > 0 and False:
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
