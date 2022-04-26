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
from tqdm import tqdm
import streamlit as st
import matplotlib
import time

from jarvis.prediction.jarvis2D import JarvisPredictor2D
import jarvis.prediction.prediction_utils as utils
from jarvis.config.project_manager import ProjectManager
from jarvis.utils.skeleton import get_skeleton



def predict2D(params):
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg

    params.output_dir = os.path.join(project.parent_dir,
                cfg.PROJECTS_ROOT_PATH, params.project_name,
                'predictions',
                f'Predictions_2D_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(params.output_dir, exist_ok = True)

    jarvisPredictor = JarvisPredictor2D(cfg, params.weights_center_detect,
                params.weights_keypoint_detect, params.trt_mode)

    cap = cv2.VideoCapture(params.recording_path)
    cap.set(1,params.frame_start)
    img_size  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    frameRate = cap.get(cv2.CAP_PROP_FPS)

    if params.make_video:
        out = cv2.VideoWriter(os.path.join(params.output_dir,
                    params.recording_path.split('/')[-1].split(".")[0] + ".mp4"),
                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frameRate,
                    (img_size[0],img_size[1]))

    #create skeleton idxs and colors for plotting
    colors, line_idxs = get_skeleton(cfg)

    csvfile = open(os.path.join(params.output_dir, 'data2D.csv'), 'w',
                newline='')
    writer = csv.writer(csvfile, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #if keypoint names are defined, add header to csvs
    if (len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS):
        create_header(writer, cfg)

    assert params.frame_start < cap.get(cv2.CAP_PROP_FRAME_COUNT), \
                "frame_start bigger than total framecount!"
    if (params.number_frames == -1):
        params.number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) \
                    - params.frame_start
    else:
        assert params.frame_start+params.number_frames <= cap.get(cv2.CAP_PROP_FRAME_COUNT), \
                    "make sure your selected segment is not longer that the " \
                    "total video!"

    for frame_num in tqdm(range(params.number_frames)):
        ret, img_orig = cap.read()
        img = torch.from_numpy(
                img_orig).cuda().float().permute(2,0,1)[[2, 1, 0]]/255.

        points2D, maxvals = jarvisPredictor(img.unsqueeze(0))

        if points2D != None:
            points2D = points2D.cpu().numpy()
            maxvals = maxvals.cpu().numpy()
            row = []
            for i,point in enumerate(points2D):
                row = row + point.tolist() + [maxvals[i]]
            writer.writerow(row)
            if params.make_video:
                for line in line_idxs:
                    utils.draw_line(img_orig, line, points2D,
                            img_size, colors[line[1]])
                for j,point in enumerate(points2D):
                    utils.draw_point(img_orig, point, img_size,
                            colors[j])

        else:
            row = []
            for i in range(keypointDetect.main_cfg.KEYPOINTDETECT.NUM_JOINTS*3):
                row = row + ['NaN']
            writer.writerow(row)

        if params.make_video:
            out.write(img_orig)
        if params.progress_bar != None:
            params.progress_bar.progress(float(frame_num+1)
                        / float(number_frames))

    if params.make_video:
        out.release()
    cap.release()


def create_header(writer, cfg):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3)
                for x in cfg.KEYPOINT_NAMES))
    coords = ['x','y','confidence']*len(cfg.KEYPOINT_NAMES)
    writer.writerow(joints)
    writer.writerow(coords)
