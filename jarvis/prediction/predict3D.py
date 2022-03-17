"""
predict3D.py
=================
Functions to run 3D inference and visualize the results
"""

import os
import csv
import itertools
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib
import json
import itertools
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

from jarvis.efficienttrack.efficienttrack import EfficientTrack
import jarvis.efficienttrack.darkpose as darkpose
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.dataset.utils import ReprojectionTool
import jarvis.prediction.prediction_utils as utils


def load_reprojection_tools(cfg, cameras_to_use = None):
    if cameras_to_use != None:
        print (f"Using subset of cameras: {cameras_to_use}.")
    dataset_dir = os.path.join(cfg.PARENT_DIR, cfg.DATASET.DATASET_ROOT_DIR, cfg.DATASET.DATASET_3D)
    dataset_json = open(os.path.join(dataset_dir, 'annotations',
                'instances_val.json'))
    data = json.load(dataset_json)
    reproTools = {}
    for calibParams in data['calibrations']:
        calibPaths = {}
        for cam in data['calibrations'][calibParams]:
            if cameras_to_use == None or cam in cameras_to_use:
                calibPaths[cam] = data['calibrations'][calibParams][cam]
        reproTools[calibParams] = ReprojectionTool(
                    dataset_dir,calibPaths)
    dataset_json.close()
    return reproTools


def get_videos_from_recording_path(recording_path, reproTool):
    videos = os.listdir(recording_path)
    video_paths = []
    for i, camera in enumerate(reproTool.cameras):
        for video in videos:
            if camera == video.split('.')[0]:
                video_paths.append(os.path.join(recording_path, video))
        assert (len(video_paths) == i+1), "Missing Recording for camera " + camera
    return video_paths


def predictPosesVideos(hybridNet, centerDetect, reproTool, recording_path,
        output_dir, frameStart = 0, numberFrames = -1, make_videos = True, skeletonPreset = None, progressBar =None):

    img_downsampled_shape = centerDetect.cfg.IMAGE_SIZE
    def read_images(cap):
        ret, img = cap.read()
        return img

    def process_images(img):
        img = ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                / 255.0 - hybridNet.cfg.DATASET.MEAN) / hybridNet.cfg.DATASET.STD)
        img = cv2.resize(img, (img_downsampled_shape,img_downsampled_shape))
        return img

    if os.path.exists(output_dir) and progressBar == None:
        print ("Output directory already exists! Override? (Y)es/(N)o")
        valid_accepts = ['yes', 'Yes', 'y', 'Y']
        valid_declines = ['no', 'No', 'n', 'N']
        got_valid_answer = False
        while not got_valid_answer:
            ans = input()
            if ans in valid_declines:
                got_valid_answer = True
                print ("Aborting prediction!")
                return
            elif ans in valid_accepts:
                got_valid_answer = True
            else:
                print ("Please enter either yes or no!")
    if (make_videos):
        os.makedirs(os.path.join(output_dir, 'Videos'), exist_ok = True)
    else:
        os.makedirs(output_dir, exist_ok = True)

    caps = []
    outs = []
    video_paths = get_videos_from_recording_path(recording_path, reproTool)
    img_size = [0,0]
    for path in video_paths:
        caps.append(cv2.VideoCapture(path))
        img_size_new = [int(caps[-1].get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(caps[-1].get(cv2.CAP_PROP_FRAME_HEIGHT))]
        assert (img_size == [0,0] or img_size == img_size_new), "All videos need to have the same resolution"
        img_size = img_size_new
        if (make_videos):
            frameRate = caps[-1].get(cv2.CAP_PROP_FPS)
            os.makedirs(os.path.join(output_dir, 'Videos'), exist_ok = True)
            outs.append(cv2.VideoWriter(os.path.join(output_dir, 'Videos',
                        path.split('/')[-1].split(".")[0] + ".avi"),
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frameRate,
                        (img_size[0],img_size[1])))

    for cap in caps:
            cap.set(1,frameStart)

    num_cameras = len(caps)
    hybridNet.cfg.DATASET.NUM_CAMERAS = num_cameras

    counter = 0
    with open(os.path.join(output_dir, 'data3D.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

        colors = []
        line_idxs = []
        if isinstance(skeletonPreset, str):
            colors, line_idxs = utils.get_colors_and_lines(skeletonPreset)
            create_header(writer, skeletonPreset, hybridNet.cfg.KEYPOINTDETECT.NUM_JOINTS)
        elif skeletonPreset != None:
            colors = skeletonPreset["colors"]
            line_idxs = skeletonPreset["line_idxs"]
        else:
            cmap = matplotlib.cm.get_cmap('jet')
            for i in range(hybridNet.cfg.KEYPOINTDETECT.NUM_JOINTS):
                colors.append(((np.array(
                        cmap(float(i)/hybridNet.cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                        255).astype(int)[:3]).tolist())

        assert frameStart < caps[0].get(cv2.CAP_PROP_FRAME_COUNT), "frameStart bigger than total framecount!"
        if (numberFrames == -1):
            numberFrames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))-frameStart
        else:
            assert frameStart+numberFrames <= caps[0].get(cv2.CAP_PROP_FRAME_COUNT), "make sure your selected segment is not longer that the total video!"
        for frame_num in tqdm(range(numberFrames)):
            imgs = []
            imgs_orig = []
            centerHMs = []
            camsToUse = []

            imgs_orig = Parallel(n_jobs=-1, require='sharedmem')(delayed(read_images)(cap) for cap in caps)
            downsampling_scale = np.array([float(imgs_orig[0].shape[1]/img_downsampled_shape), float(imgs_orig[0].shape[0]/img_downsampled_shape)])
            imgs = Parallel(n_jobs=-1, require='sharedmem')(delayed(process_images, )(img) for img in imgs_orig)

            imgs = torch.from_numpy(np.array(imgs).transpose(0,3,1,2)).cuda().float()
            outputs = centerDetect.model(imgs)
            preds, maxvals = darkpose.get_final_preds(outputs[1].clamp(0,255).detach().cpu().numpy(), None)
            camsToUse = []

            for i,val in enumerate(maxvals[:]):
                if val > 100:
                    camsToUse.append(i)
            if len(camsToUse) >= 2:
                center3D = torch.from_numpy(reproTool.reconstructPoint((preds.reshape(num_cameras,2)*(downsampling_scale*2)).transpose(), camsToUse))
                reproPoints = reproTool.reprojectPoint(center3D)


                errors = []
                errors_valid = []
                for i in range(num_cameras):
                    if maxvals[i] > 100:
                        errors.append(np.linalg.norm(preds.reshape(num_cameras,2)[i]*downsampling_scale*2-reproPoints[i]))
                        errors_valid.append(np.linalg.norm(preds.reshape(num_cameras,2)[i]*downsampling_scale*2-reproPoints[i]))
                    else:
                        errors.append(0)
                medianError = np.median(np.array(errors_valid))
                camsToUse = []
                for i,val in enumerate(maxvals[:]):
                    if val > 100 and errors[i] < 2*medianError:
                        camsToUse.append(i)
                center3D = torch.from_numpy(reproTool.reconstructPoint((preds.reshape(num_cameras,2)*downsampling_scale*2).transpose(), camsToUse))
                reproPoints = reproTool.reprojectPoint(center3D)
                imgs = []
                bbox_hw = int(hybridNet.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
                for idx,reproPoint in enumerate(reproPoints):
                    reproPoint = reproPoint.astype(int)
                    reproPoints[idx][0] = min(max(reproPoint[0], bbox_hw), img_size[0]-1-bbox_hw)
                    reproPoints[idx][1] = min(max(reproPoint[1], bbox_hw), img_size[1]-1-bbox_hw)
                    reproPoint[0] = reproPoints[idx][0]
                    reproPoint[1] = reproPoints[idx][1]
                    img = imgs_orig[idx][reproPoint[1]-bbox_hw:reproPoint[1]+bbox_hw, reproPoint[0]-bbox_hw:reproPoint[0]+bbox_hw, :]
                    #cv2.imshow("", img)
                    #cv2.waitKey(0)
                    img = ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0-hybridNet.cfg.DATASET.MEAN)/hybridNet.cfg.DATASET.STD)
                    imgs.append(img)


                imgs = torch.from_numpy(np.array(imgs))
                imgs = imgs.permute(0,3,1,2).view(1,num_cameras,3,hybridNet.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,hybridNet.cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda().float()
                centerHMs = np.array(reproPoints).astype(int)

            if len(camsToUse) >= 2:
                center3D = center3D.int().cuda()
                centerHMs = torch.from_numpy(centerHMs).cuda()
                heatmap3D, heatmaps_padded, points3D_net = hybridNet.model(imgs,
                            torch.tensor(img_size).cuda(),
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

                points2D = np.array(points2D)
                if make_videos:
                    for i in range(len(outs)):
                        for line in line_idxs:
                            utils.draw_line(imgs_orig[i], line, points2D[:,i],
                                    img_size, colors[line[1]])
                        for j,points in enumerate(points2D):
                            utils.draw_point(imgs_orig[i], points[i], img_size,
                                    colors[j])

            else:
                row = []
                for i in range(hybridNet.cfg.KEYPOINTDETECT.NUM_JOINTS*3):
                    row = row + ['NaN']
                writer.writerow(row)
            if make_videos:
                for i,out in enumerate(outs):
                    out.write(imgs_orig[i])
                if progressBar != None:
                    progressBar.progress(float(frame_num+1)/float(numberFrames))

        if make_videos:
            for out in outs:
                out.release()
        for cap in caps:
            cap.release()


def create_header(writer, skeletonPreset, num_keypoints):
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
    header2 = ['x','y','z']*num_keypoints
    writer.writerow(header2)
