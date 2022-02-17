"""
predict2D.py
=================
Functions to run 2D inference and visualize the results
"""

import os
import csv
import itertools
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import streamlit as st
import matplotlib


from jarvis.efficienttrack.efficienttrack import EfficientTrack
import jarvis.efficienttrack.darkpose as darkpose
import jarvis.prediction.prediction_utils as utils


def predictPosesVideo(keypointDetect, centerDetect, video_path,
            output_dir, frameStart = 0, numberFrames = -1,
            make_video = True, skeletonPreset = None, progressBar = None):
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
    os.makedirs(output_dir, exist_ok = True)

    cap = cv2.VideoCapture(video_path)
    cap.set(1,frameStart)
    img_size  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    frameRate = cap.get(cv2.CAP_PROP_FPS)

    if make_video:
        out = cv2.VideoWriter(os.path.join(output_dir,
                    video_path.split('/')[-1].split(".")[0] + ".avi"),
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frameRate,
                    (img_size[0],img_size[1]))

    with open(os.path.join(output_dir, 'data2D.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
        colors = []
        line_idxs = []
        if isinstance(skeletonPreset, str):
            colors, line_idxs = utils.get_colors_and_lines(skeletonPreset)
            create_header(writer, skeletonPreset, keypointDetect.main_cfg.KEYPOINTDETECT.NUM_JOINTS)
        elif skeletonPreset != None:
            colors = skeletonPreset["colors"]
            line_idxs = skeletonPreset["line_idxs"]
        else:
            cmap = matplotlib.cm.get_cmap('jet')
            for i in range(keypointDetect.main_cfg.KEYPOINTDETECT.NUM_JOINTS):
                colors.append(((np.array(
                        cmap(float(i)/keypointDetect.main_cfg.KEYPOINTDETECT.NUM_JOINTS)) *
                        255).astype(int)[:3]).tolist())

        if (numberFrames == -1):
            numberFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in tqdm(range(numberFrames)):
            ret, img_orig = cap.read()
            image_size_center = centerDetect.cfg.IMAGE_SIZE
            img_downsampled_shape = (image_size_center, image_size_center)
            downsampling_scale = np.array(
                        [float(img_orig.shape[1]/image_size_center),
                         float(img_orig.shape[0]/image_size_center)])
            img = ((cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB).astype(np.float32)
                    / 255.0 - keypointDetect.main_cfg.DATASET.MEAN) / keypointDetect.main_cfg.DATASET.STD)
            img = cv2.resize(img, img_downsampled_shape)

            img = torch.from_numpy(img.transpose(2,0,1)).cuda().float()
            outputs = centerDetect.model(torch.unsqueeze(img,0))
            center, maxval = darkpose.get_final_preds(
                        outputs[1].clamp(0,255).detach().cpu().numpy(), None)
            center = center.squeeze()
            center =center*(downsampling_scale*2)
            bbox_hw = int(keypointDetect.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE/2)
            center[0] = min(max(center[0], bbox_hw), img_size[0]-1-bbox_hw)
            center[1] = min(max(center[1], bbox_hw), img_size[1]-1-bbox_hw)
            center = center.astype(int)

            img = img_orig[center[1]-bbox_hw:center[1]+bbox_hw, center[0]-bbox_hw:center[0]+bbox_hw, :]
            img = ((cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) /
                        255.0-keypointDetect.main_cfg.DATASET.MEAN)/keypointDetect.main_cfg.DATASET.STD)

            img = torch.from_numpy(img)
            img = img.permute(2,0,1).view(1,3,
                        keypointDetect.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE,
                        keypointDetect.main_cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE).cuda().float()
            if maxval >= 50:
                outputs = keypointDetect.model(img)
                preds, maxvals = darkpose.get_final_preds(
                            outputs[1].clamp(0,255).detach().cpu().numpy(), None)
                preds = preds.squeeze()
                preds = preds*2+center-bbox_hw
                row = []
                for i,point in enumerate(preds.squeeze()):
                    row = row + point.tolist() + [maxvals[0,i]]
                writer.writerow(row)
                if make_video:
                    assert (preds.shape[0] <= len(colors)), "colorPreset does not match number of Keypoints!"
                    for line in line_idxs:
                        utils.draw_line(img_orig, line, preds,
                                img_size, colors[line[1]])
                    for j,point in enumerate(preds):
                        if maxvals[0,j] > 50:
                            utils.draw_point(img_orig, point, img_size,
                                    colors[j])
            else:
                row = []
                for i in range(keypointDetect.main_cfg.KEYPOINTDETECT.NUM_JOINTS*3):
                    row = row + ['NaN']
                writer.writerow(row)

            if make_video:
                out.write(img_orig)
            if progressBar != None:
                progressBar.progress(float(frame_num+1)/float(numberFrames))

        if make_video:
            out.release()
        cap.release()
        del centerDetect
        del keypointDetect


def create_header(writer, skeletonPreset, num_keypoints):
    if skeletonPreset == "Hand":
        joints = ["Pinky_T","Pinky_D","Pinky_M","Pinky_P","Ring_T",
                  "Ring_D","Ring_M","Ring_P","Middle_T","Middle_D",
                  "Middle_M","Middle_P","Index_T","Index_D","Index_M",
                  "Index_P","Thumb_T","Thumb_D","Thumb_M","Thumb_P",
                  "Palm", "Wrist_U","Wrist_R"]
        assert (num_keypoints == len(joints)), "Number of keypoints does not match hand preset"
        header = []
        joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in joints))
        header = header + joints
        writer.writerow(header)
    header2 = ['x','y','confidence']*num_keypoints
    writer.writerow(header2)
