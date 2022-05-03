"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import torch

from jarvis.utils.reprojection import get_repro_tool
from jarvis.config.project_manager import ProjectManager
from jarvis.utils.skeleton import get_skeleton
import jarvis.visualization.visualization_utils as utils


def create_videos3D(params):
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg
    reproTool = get_repro_tool(cfg, params.dataset_name, 'cpu')

    params.output_dir = os.path.join(project.parent_dir,
                cfg.PROJECTS_ROOT_PATH, params.project_name,
                'visualization',
                f'Videos_3D_{time.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(params.output_dir, exist_ok = True)

    #create openCV video read and write streams
    video_paths, make_video_index = get_video_paths_and_cam_index(
                params.recording_path, reproTool, params.video_cam_list)

    caps, outs, img_size = create_video_writer_and_reader(params, reproTool,
                video_paths, make_video_index)

    colors, line_idxs = get_skeleton(cfg)
    data = np.genfromtxt(params.data_csv, delimiter=',')
    if np.isnan(data[0,0]):
        data = data[2:]
    points3D = np.delete(data, list(range(3, data.shape[1], 4)), axis=1)
    confidences = data[:, 3::4]


    if (params.number_frames == -1):
        params.number_frames = (int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
                    - params.frame_start)
    else:
        assert params.frame_start+params.number_frames \
                    <= caps[0].get(cv2.CAP_PROP_FRAME_COUNT), \
                    "make sure your selected segment is not " \
                    "longer that the total video!"

    imgs_orig = np.zeros((len(caps), img_size[1],
                img_size[0], 3)).astype(np.uint8)

    for frame_num in tqdm(range(params.number_frames)):
        Parallel(n_jobs=12, require='sharedmem')(delayed(read_images)
                    (cap, slice, imgs_orig) for slice, cap in enumerate(caps))
        points3D_net = torch.from_numpy(
                    points3D[frame_num].reshape(-1,3)).float()
        confidence = confidences[frame_num]

        if points3D_net != None:
            points2D = reproTool.reprojectPoint(
                        points3D_net).numpy()

            points2D = np.array(points2D)
            for i in range(len(outs)):
                if make_video_index[i]:
                    for line in line_idxs:
                        utils.draw_line(imgs_orig[i], line, points2D[:,i],
                                img_size, colors[line[1]])
                    for j,points in enumerate(points2D):
                        utils.draw_point(imgs_orig[i], points[i], img_size,
                                colors[j])
        for i,out in enumerate(outs):
            if make_video_index[i]:
                out.write(imgs_orig[i])
        if params.progress_bar != None:
            params.progress_bar.progress(float(frame_num+1)
                        / float(params.number_frames))

    for i,out in enumerate(outs):
        if make_video_index[i]:
            out.release()
    for cap in caps:
        cap.release()


def get_video_paths_and_cam_index(recording_path, reproTool, video_cam_list):
    videos = os.listdir(recording_path)
    video_paths = []
    make_video_index = []
    for i, camera in enumerate(reproTool.cameras):
        for video in videos:
            if camera == video.split('.')[0]:
                video_paths.append(os.path.join(recording_path, video))
                make_video_index.append(camera in video_cam_list)
        assert (len(video_paths) == i+1), \
                    "Missing Recording for camera " + camera
    return video_paths, make_video_index


def create_video_writer_and_reader(params, reproTool, video_paths,
            make_video_index):
    caps = []
    outs = []
    img_size = [0,0]
    for i,path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        img_size_new = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        assert (img_size == [0,0] or img_size == img_size_new), \
                    "All videos need to have the same resolution"
        img_size = img_size_new
        assert params.frame_start < cap.get(cv2.CAP_PROP_FRAME_COUNT), \
                    "frame_start bigger than total framecount!"
        cap.set(1,params.frame_start)
        caps.append(cap)
        if make_video_index[i]:
            frameRate = cap.get(cv2.CAP_PROP_FPS)
            outs.append(cv2.VideoWriter(os.path.join(params.output_dir,
                        path.split('/')[-1].split(".")[0] + ".mp4"),
                        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                        frameRate,
                        (img_size[0],img_size[1])))
        else:
            outs.append(None)

    return caps, outs, img_size


def read_images(cap, slice, imgs):
    ret, img = cap.read()
    imgs[slice] = img.astype(np.uint8)
