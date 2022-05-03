"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import os
import time
import cv2
import numpy as np
from tqdm import tqdm

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.skeleton import get_skeleton
import jarvis.visualization.visualization_utils as utils


def create_videos2D(params):
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg

    params.output_dir = os.path.join(project.parent_dir,
                cfg.PROJECTS_ROOT_PATH, params.project_name,
                'visualization',
                f'Videos_2D_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(params.output_dir, exist_ok = True)

    cap = cv2.VideoCapture(params.recording_path)
    cap.set(1,params.frame_start)
    img_size  = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    frameRate = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(os.path.join(params.output_dir,
                params.recording_path.split('/')[-1].split(".")[0] + ".mp4"),
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frameRate,
                (img_size[0],img_size[1]))

    colors, line_idxs = get_skeleton(cfg)
    points2D_all = np.genfromtxt(params.data_csv, delimiter=',')
    if np.isnan(points2D_all[0,0]):
        points2D_all = points2D_all[2:]

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
        points2D = points2D_all[frame_num].reshape(-1,3)

        if not np.isnan(points2D[0,0]):
            for line in line_idxs:
                utils.draw_line(img_orig, line, points2D,
                        img_size, colors[line[1]])
            for j,point in enumerate(points2D):
                utils.draw_point(img_orig, point, img_size,
                        colors[j])

        out.write(img_orig)
        if params.progress_bar != None:
            params.progress_bar.progress(float(frame_num+1)
                        / float(params.number_frames))

    out.release()
    cap.release()
