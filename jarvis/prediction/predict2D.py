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
from tqdm import tqdm
import streamlit as st
import time
from ruamel.yaml import YAML

from jarvis.prediction.jarvis2D import JarvisPredictor2D
from jarvis.config.project_manager import ProjectManager


def create_info_file(params):
    with open(os.path.join(params.output_dir, 'info.yaml'), 'w') as file:
        yaml=YAML()
        yaml.dump({'recording_path': params.recording_path,
                    'frame_start': params.frame_start,
                    'number_frames': params.number_frames}, file)


def predict2D(params):
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg

    params.output_dir = os.path.join(project.parent_dir,
                cfg.PROJECTS_ROOT_PATH, params.project_name,
                'predictions', 'predictions2D',
                f'Predictions_2D_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(params.output_dir, exist_ok = True)
    create_info_file(params)

    jarvisPredictor = JarvisPredictor2D(cfg, params.weights_center_detect,
                params.weights_keypoint_detect, params.trt_mode)

    cap = cv2.VideoCapture(params.recording_path)
    cap.set(1,params.frame_start)
    img_size  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]

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
        assert params.frame_start+params.number_frames \
                    <= cap.get(cv2.CAP_PROP_FRAME_COUNT), \
                    "make sure your selected segment is not longer that the " \
                    "total video!"

    for frame_num in tqdm(range(params.number_frames)):
        ret, img_orig = cap.read()
        img = torch.from_numpy(
                img_orig).cuda().float().permute(2,0,1)[[2, 1, 0]]/255.

        points2D, confidences = jarvisPredictor(img.unsqueeze(0))

        if points2D != None:
            points2D = points2D.cpu().numpy()
            confidences = confidences.cpu().numpy()
            row = []
            for i,point in enumerate(points2D):
                row = row + point.tolist() + [confidences[i]]
            writer.writerow(row)

        else:
            row = []
            for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS*3):
                row = row + ['NaN']
            writer.writerow(row)


        if params.progress_bar != None:
            params.progress_bar.progress(float(frame_num+1)
                        / float(params.number_frames))

    cap.release()


def create_header(writer, cfg):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3)
                for x in cfg.KEYPOINT_NAMES))
    coords = ['x','y','confidence']*len(cfg.KEYPOINT_NAMES)
    writer.writerow(joints)
    writer.writerow(coords)
