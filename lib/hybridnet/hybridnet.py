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
            self.model.load_state_dict(state_dict, strict=True)
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


    def predictPosesVideos(self, centerDetect, reproTool, cameraMatrices,video_paths, output_dir, frameStart = 0, frameEnd = -1, makeVideos = True):
        import itertools
        from joblib import Parallel, delayed
        import cv2
        import csv

        def process(cap):
            ret, img = cap.read()
            return img

        caps = []
        outs = []
        for path in video_paths:
            caps.append(cv2.VideoCapture(path))
            os.makedirs(os.path.join(output_dir, 'Videos'), exist_ok = True)
            img_size = self.cfg.DATASET.IMAGE_SIZE
            outs.append(cv2.VideoWriter(os.path.join(output_dir, 'Videos',
                        path.split('/')[-1].split(".")[0] + ".avi"),
                        cv2.VideoWriter_fourcc('M','J', 'P', 'G'), 100,
                        (img_size[0],img_size[1])))

        for cap in caps:
                cap.set(1,frameStart) #99640

        counter = 0
        with open(os.path.join(output_dir, 'data3D.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # header = ['Frame Idx']
            # joints = ["Pinky_T","Pinky_D","Pinky_M","Pinky_P","Ring_T","Ring_D","Ring_M","Ring_P","Middle_T","Middle_D","Middle_M","Middle_P","Index_T","Index_D","Index_M","Index_P","Thumb_T","Thumb_D","Thumb_M","Thumb_P","Palm", "Wrist_U","Wrist_R"]
            # joints = joints
            # joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in joints))
            # header = header + joints
            # header2 = [""]
            # header2 = header2 + ['x','y', 'z']*23
            # writer.writerow(header)
            # writer.writerow(header2)
            ret = True
            while ret and (frameEnd == -1 or counter < frameEnd):
                if counter % 100 == 0:
                    print ("Analysing Frame ", counter)
                counter += 1
                imgs = []
                imgs_orig = []
                centerHMs = []
                camsToUse = []

                imgs_orig = Parallel(n_jobs=-1, require='sharedmem')(delayed(process)(cap) for cap in caps)
                img_downsampled_shape = (int(imgs_orig[0].shape[1]/self.cfg.CENTERDETECT.DOWNSAMPLING_FACTOR),int(imgs_orig[0].shape[0]/self.cfg.CENTERDETECT.DOWNSAMPLING_FACTOR))
                imgs = torch.zeros(self.cfg.DATASET.NUM_CAMERAS,3,img_downsampled_shape[1],img_downsampled_shape[0])
                for i,img in enumerate(imgs_orig[:]):
                    img = ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0-self.cfg.DATASET.MEAN)/self.cfg.DATASET.STD)
                    img = cv2.resize(img, img_downsampled_shape)
                    imgs[i] = torch.from_numpy(cv2.resize(img, img_downsampled_shape).transpose(2,0,1))

                imgs = imgs.cuda()
                outputs = centerDetect.model(imgs)
                preds, maxvals = darkpose.get_final_preds(outputs[1].clamp(0,255).detach().cpu().numpy(), None)
                camsToUse = []

                for i,val in enumerate(maxvals[:]):
                    if val > 150:
                        camsToUse.append(i)
                print (len(camsToUse))
                if len(camsToUse) >= 2:
                    center3D = torch.from_numpy(reproTool.reconstructPoint((preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)*(self.cfg.CENTERDETECT.DOWNSAMPLING_FACTOR*2)).transpose(), camsToUse))
                    reproPoints = reproTool.reprojectPoint(center3D)

                    errors = []
                    errors_valid = []
                    for i in range(self.cfg.DATASET.NUM_CAMERAS):
                        if maxvals[i] > 180:
                            errors.append(np.linalg.norm(preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)[i]*8-reproPoints[i]))
                            errors_valid.append(np.linalg.norm(preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)[i]*8-reproPoints[i]))
                        else:
                            errors.append(0)
                    medianError = np.median(np.array(errors_valid))
                    # print ("Error: ", medianError)
                    # print ("Var based: ", medianError+2*np.sqrt(np.var(errors_valid)),medianError*4)
                    camsToUse = []
                    for i,val in enumerate(maxvals[:]):
                        if val > 180 and errors[i] < 2*medianError:
                            camsToUse.append(i)
                    center3D = torch.from_numpy(reproTool.reconstructPoint((preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)*8).transpose(), camsToUse))
                    reproPoints = reproTool.reprojectPoint(center3D)
                    print (center3D)
                imgs = []
                bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
                for idx,reproPoint in enumerate(reproPoints):
                    reproPoint = reproPoint.astype(int)
                    img = imgs_orig[idx][reproPoint[1]-bbox_hw:reproPoint[1]+bbox_hw, reproPoint[0]-bbox_hw:reproPoint[0]+bbox_hw, :]
                    img = ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0-self.cfg.DATASET.MEAN)/self.cfg.DATASET.STD)
                    imgs.append(img)

                if not ret:
                    break
                imgs = torch.from_numpy(np.array(imgs))
                imgs = imgs.permute(0,3,1,2).view(1,self.cfg.DATASET.NUM_CAMERAS,3,self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda().float()
                centerHMs = np.array(reproPoints).astype(int)
                if len(camsToUse) >= 2:
                    center3D = center3D.int().cuda()
                    centerHMs = torch.from_numpy(centerHMs).cuda()
                    heatmap3D, heatmaps_padded, points3D_net = self.model(imgs, torch.unsqueeze(centerHMs,0), torch.unsqueeze(center3D, 0), torch.unsqueeze(cameraMatrices.cuda(),0))
                    row = [counter]
                    for point in points3D_net.squeeze():
                        row = row + point.tolist()
                    writer.writerow(row)
                    points2D = []
                    for point in points3D_net[0].cpu().numpy():
                        points2D.append(reproTool.reprojectPoint(point))


                else:
                    row = [counter]
                    for i in range(23*3):
                        row = row + ['NaN']
                    writer.writerow(row)

                colors = [(255,0,0), (255,0,0),(255,0,0),(255,0,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(255,255,0),(255,255,0),(255,255,0), (255,255,0),
                (0,255,255),(0,255,255),(0,255,255),(0,255,255), (255,0,255),(100,0,100),(100,0,100)]
                for i,out in enumerate(outs):
                    img = imgs_orig[i]
                    if len(camsToUse) >= 2:
                        for j,point in enumerate(points2D):
                            cv2.circle(img, (int(point[i][0]), int(point[i][1])), 2, colors[j], thickness=5)
                    out.write(img)

            for out in outs:
                out.release()
            for cap in caps:
                cap.release()
