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
import itertools
from joblib import Parallel, delayed
import cv2
import csv
import matplotlib

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


                self.optimizer.zero_grad()
                if self.cfg.USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs, centerHM, center3D,
                                    cameraMatrices, intrinsicMatrices, distortionCoefficients)
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
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                else:
                    outputs = self.model(imgs, centerHM, center3D,
                                         cameraMatrices, intrinsicMatrices, distortionCoefficients)
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



                        if self.cfg.USE_MIXED_PRECISION:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(imgs, centerHM, center3D,
                                                     cameraMatrices, intrinsicMatrices, distortionCoefficients)
                                loss = self.criterion(outputs[0], heatmap3D)
                                loss = loss.mean()
                                acc = torch.mean(torch.sqrt(torch.sum(
                                        (keypoints-outputs[2])**2, dim = 2)))
                        else:
                            outputs = self.model(imgs, centerHM, center3D,
                                                 cameraMatrices,intrinsicMatrices, distortionCoefficients)
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


    def predictPosesVideos(self, centerDetect, reproTool, video_paths,
            output_dir, frameStart = 0, numberFrames = -1, skeletonPreset = None):
        def read_images(cap):
            ret, img = cap.read()
            return img

        def process_images(img):
            img = ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                    / 255.0 - self.cfg.DATASET.MEAN) / self.cfg.DATASET.STD)
            img = cv2.resize(img, img_downsampled_shape)
            return img

        img_size = self.cfg.DATASET.IMAGE_SIZE

        caps = []
        outs = []
        for path in video_paths:
            caps.append(cv2.VideoCapture(path))
            frameRate = caps[-1].get(cv2.CAP_PROP_FPS)
            os.makedirs(os.path.join(output_dir, 'Videos'), exist_ok = True)
            outs.append(cv2.VideoWriter(os.path.join(output_dir, 'Videos',
                        path.split('/')[-1].split(".")[0] + ".avi"),
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frameRate,
                        (img_size[0],img_size[1])))

        for cap in caps:
                cap.set(1,frameStart)
        counter = 0
        with open(os.path.join(output_dir, 'data3D.csv'), 'w', newline='') as csvfile:
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
                for i in range(self.cfg.KEYPOINTDETECT.NUM_JOINTS):
                    colors.append(((np.array(
                            cmap(float(i)/self.cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                            255).astype(int)[:3]).tolist())

            ret = True
            if (numberFrames == -1):
                numberFrames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
            while ret and (numberFrames == -1 or counter < numberFrames):
                if counter % 100 == 0:
                    print ("Analysing Frame {}/{}".format(counter, numberFrames))
                counter += 1
                imgs = []
                imgs_orig = []
                centerHMs = []
                camsToUse = []

                imgs_orig = Parallel(n_jobs=-1, require='sharedmem')(delayed(read_images)(cap) for cap in caps)
                img_downsampled_shape = (self.cfg.CENTERDETECT.IMAGE_SIZE, self.cfg.CENTERDETECT.IMAGE_SIZE)
                downsampling_scale = np.array([float(imgs_orig[0].shape[1]/self.cfg.CENTERDETECT.IMAGE_SIZE), float(imgs_orig[0].shape[0]/self.cfg.CENTERDETECT.IMAGE_SIZE)])
                imgs = Parallel(n_jobs=-1, require='sharedmem')(delayed(process_images)(img) for img in imgs_orig)

                imgs = torch.from_numpy(np.array(imgs).transpose(0,3,1,2)).cuda().float()
                outputs = centerDetect.model(imgs)
                preds, maxvals = darkpose.get_final_preds(outputs[1].clamp(0,255).detach().cpu().numpy(), None)
                camsToUse = []

                for i,val in enumerate(maxvals[:]):
                    if val > 100:
                        camsToUse.append(i)
                if len(camsToUse) >= 2:
                    center3D = torch.from_numpy(reproTool.reconstructPoint((preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)*(downsampling_scale*2)).transpose(), camsToUse))
                    reproPoints = reproTool.reprojectPoint(center3D)


                    errors = []
                    errors_valid = []
                    for i in range(self.cfg.DATASET.NUM_CAMERAS):
                        if maxvals[i] > 100:
                            errors.append(np.linalg.norm(preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)[i]*downsampling_scale*2-reproPoints[i]))
                            errors_valid.append(np.linalg.norm(preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)[i]*downsampling_scale*2-reproPoints[i]))
                        else:
                            errors.append(0)
                    medianError = np.median(np.array(errors_valid))
                    camsToUse = []
                    for i,val in enumerate(maxvals[:]):
                        if val > 100 and errors[i] < 2*medianError:
                            camsToUse.append(i)
                    center3D = torch.from_numpy(reproTool.reconstructPoint((preds.reshape(self.cfg.DATASET.NUM_CAMERAS,2)*downsampling_scale*2).transpose(), camsToUse))
                    reproPoints = reproTool.reprojectPoint(center3D)
                imgs = []
                bbox_hw = int(self.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
                for idx,reproPoint in enumerate(reproPoints):
                    reproPoint = reproPoint.astype(int)
                    reproPoints[idx][0] = min(max(reproPoint[0], bbox_hw), img_size[0]-1-bbox_hw)
                    reproPoints[idx][1] = min(max(reproPoint[1], bbox_hw), img_size[1]-1-bbox_hw)
                    reproPoint[0] = reproPoints[idx][0]
                    reproPoint[1] = reproPoints[idx][1]
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
                    heatmap3D, heatmaps_padded, points3D_net = self.model(imgs,
                                torch.unsqueeze(centerHMs,0),
                                torch.unsqueeze(center3D, 0),
                                torch.unsqueeze(reproTool.cameraMatrices.cuda(),0),
                                torch.unsqueeze(reproTool.intrinsicMatrices.cuda(),0),
                                torch.unsqueeze(reproTool.distortionCoefficients.cuda(),0))
                    row = []
                    for point in points3D_net.squeeze():
                        row = row + point.tolist()
                    writer.writerow(row)
                    points2D = []
                    for point in points3D_net[0].cpu().numpy():
                        points2D.append(reproTool.reprojectPoint(point))

                    for i in range(len(outs)):
                        for line in line_idxs:
                            self.draw_line(imgs_orig[i], line, points2D,
                                    img_size, colors[line[1]], i)
                        for j,point in enumerate(points2D):
                            self.draw_point(imgs_orig[i], point, img_size,
                                    colors[j], i)

                else:
                    row = []
                    for i in range(23*3):
                        row = row + ['NaN']
                    writer.writerow(row)

                for i,out in enumerate(outs):
                    out.write(imgs_orig[i])

            for out in outs:
                out.release()
            for cap in caps:
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
        header2 = ['x','y','z']*self.cfg.KEYPOINTDETECT.NUM_JOINTS
        writer.writerow(header2)


    def draw_line(self, img, line, points2D, img_size, color, i):
        array_sum = np.sum(np.array(points2D))
        array_has_nan = np.isnan(array_sum)
        if ((not array_has_nan) and int(points2D[line[0]][i][0]) < img_size[0]-1
                and int(points2D[line[0]][i][0]) > 0
                and  int(points2D[line[1]][i][0]) < img_size[0]-1
                and int(points2D[line[1]][i][0]) > 0
                and int(points2D[line[0]][i][1]) < img_size[1]-1
                and int(points2D[line[0]][i][1]) > 0
                and int(points2D[line[1]][i][1]) < img_size[1]-1
                and int(points2D[line[1]][i][1]) > 0):
            cv2.line(img,
                    (int(points2D[line[0]][i][0]), int(points2D[line[0]][i][1])),
                    (int(points2D[line[1]][i][0]), int(points2D[line[1]][i][1])),
                    color, 1)


    def draw_point(self, img, point, img_size, color, i):
        array_sum = np.sum(np.array(point))
        array_has_nan = np.isnan(array_sum)
        if ((not array_has_nan) and (point[i][0] < img_size[0]-1
                and point[i][0] > 0 and point[i][1] < img_size[1]-1
                and point[i][1] > 0)):
            cv2.circle(img, (int(point[i][0]), int(point[i][1])),
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
