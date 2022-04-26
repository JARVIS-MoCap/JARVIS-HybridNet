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
import matplotlib
import json
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import time

from jarvis.utils.reprojection import ReprojectionTool, load_reprojection_tools
from jarvis.utils.skeleton import get_skeleton
from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D import JarvisPredictor3D
import jarvis.prediction.prediction_utils as utils



def predict3D(params):
    #Load project and config
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg

    jarvisPredictor = JarvisPredictor3D(cfg, params.weights_center_detect,
                params.weights_hybridnet, params.trt_mode)

    reproTool = get_repro_tool(cfg, params.dataset_name)

    params.output_dir = os.path.join(project.parent_dir,
                cfg.PROJECTS_ROOT_PATH, params.project_name,
                'predictions',
                f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}')
    if (params.make_videos):
        os.makedirs(os.path.join(params.output_dir, 'Videos'), exist_ok = True)
    else:
        os.makedirs(params.output_dir, exist_ok = True)


    #create openCV video read and write streams
    video_paths, make_video_index = get_videos_paths_and_cam_index(
                params.recording_path, reproTool, params.video_cam_list)
    caps, outs, img_size = create_video_writer_and_reader(params, reproTool,
                video_paths, make_video_index)

    #create skeleton idxs and colors for plotting
    colors, line_idxs = get_skeleton(cfg)

    if (params.number_frames == -1):
        params.number_frames = (int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
                    - params.frame_start)
    else:
        assert params.frame_start+params.number_frames \
                    <= caps[0].get(cv2.CAP_PROP_FRAME_COUNT), \
                    "make sure your selected segment is not " \
                    "longer that the total video!"


    csvfile = open(os.path.join(params.output_dir, 'data3D.csv'), 'w',
                newline='')
    writer = csv.writer(csvfile, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #if keypoint names are defined, add header to csvs
    if (len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS):
        create_header(writer, cfg)

    imgs_orig = np.zeros((len(caps), img_size[1],
                img_size[0], 3)).astype(np.uint8)

    for frame_num in tqdm(range(params.number_frames)):
        #load a batch of images from all cameras in parallel using joblib
        Parallel(n_jobs=12, require='sharedmem')(delayed(read_images)
                    (cap, slice, imgs_orig) for slice, cap in enumerate(caps))
        imgs = torch.from_numpy(
                imgs_orig).cuda().float().permute(0,3,1,2)[:, [2, 1, 0]]/255.

        points3D_net = jarvisPredictor(imgs,
                    reproTool.cameraMatrices.cuda(),
                    reproTool.intrinsicMatrices.cuda(),
                    reproTool.distortionCoefficients.cuda())

        if points3D_net != None:
            row = []
            for point in points3D_net.squeeze():
                row = row + point.tolist()
            writer.writerow(row)
            points2D = reproTool.reprojectPoint(
                        points3D_net.squeeze()).cpu().numpy()

            points2D = np.array(points2D)
            if params.make_videos:
                for i in range(len(outs)):
                    if make_video_index[i]:
                        for line in line_idxs:
                            utils.draw_line(imgs_orig[i], line, points2D[:,i],
                                    img_size, colors[line[1]])
                        for j,points in enumerate(points2D):
                            utils.draw_point(imgs_orig[i], points[i], img_size,
                                    colors[j])
        else:
            row = []
            for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS*3):
                row = row + ['NaN']
            writer.writerow(row)

        if params.make_videos:
            for i,out in enumerate(outs):
                if make_video_index[i]:
                    out.write(imgs_orig[i])

        if params.progress_bar != None:
            params.progress_bar.progress(float(frame_num+1)
                        / float(params.number_frames))


    if params.make_videos:
        for i,out in enumerate(outs):
            if make_video_index[i]:
                out.release()
    for cap in caps:
        cap.release()
    csvfile.close()


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
        if params.make_videos:
            if make_video_index[i]:
                frameRate = cap.get(cv2.CAP_PROP_FPS)
                outs.append(cv2.VideoWriter(os.path.join(params.output_dir,
                            'Videos',
                            path.split('/')[-1].split(".")[0] + ".mp4"),
                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                            frameRate,
                            (img_size[0],img_size[1])))
            else:
                outs.append(None)

    return caps, outs, img_size


def get_repro_tool(cfg, dataset_name):
    reproTools = load_reprojection_tools(cfg)
    if dataset_name != None and not dataset_name in reproTools:
        if os.path.isdir(dataset_name):
            dataset_dir = os.path.join(cfg.PARENT_DIR,
                        cfg.DATASET.DATASET_ROOT_DIR,
                        cfg.DATASET.DATASET_3D)
            dataset_json = open(os.path.join(dataset_dir, 'annotations',
                        'instances_val.json'))
            data = json.load(dataset_json)
            calibPaths = {}
            calibParams = list(data['calibrations'].keys())[0]
            for cam in data['calibrations'][calibParams]:
                calibPaths[cam] = \
                        data['calibrations'][calibParams][cam].split("/")[-1]
            reproTool = ReprojectionTool(dataset_name, calibPaths)
        else:
            print (f'{CLIColors.FAIL}Could not load reprojection Tool for'
                        f'specified project...{CLIColors.ENDC}')
            return None
    elif len(reproTools) == 1:
        reproTool = reproTools[list(reproTools.keys())[0]]
    elif len(reproTools) > 1:
        if dataset_name == None:
            reproTool = reproTools[list(reproTools.keys())[0]]
        else:
            reproTool = reproTools[dataset_name]
    else:
        print (f'{CLIColors.FAIL}Could not load reprojection Tool for specified'
                    f' project...{CLIColors.ENDC}')
        return None
    return reproTool


def get_videos_paths_and_cam_index(recording_path, reproTool, video_cam_list):
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


def read_images(cap, slice, imgs):
    ret, img = cap.read()
    imgs[slice] = img.astype(np.uint8)


def create_header(writer, cfg):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 3)
                for x in cfg.KEYPOINT_NAMES))
    coords = ['x','y','z']*len(cfg.KEYPOINT_NAMES)
    writer.writerow(joints)
    writer.writerow(coords)
