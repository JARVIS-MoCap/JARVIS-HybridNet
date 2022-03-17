import os
import torch
import time
import streamlit as st

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.prediction.predict2D import predictPosesVideo
from jarvis.prediction.predict3D import predictPosesVideos, load_reprojection_tools


def predict2D(project_name, video_path, weights_center_detect,
            weights_keypoint_detect, frame_start, number_frames,
            make_video, skeleton_preset, progressBar = None):
    project = ProjectManager()
    if not project.load(project_name):
        return
    centerDetect = EfficientTrack('CenterDetectInference', project.cfg,
                weights_center_detect)
    keypointDetect = EfficientTrack('KeypointDetectInference', project.cfg,
                weights_keypoint_detect)
    output_dir = os.path.join(project.parent_dir,
                project.cfg.PROJECTS_ROOT_PATH, project_name,
                'predictions', f'Predictions_2D_{time.strftime("%Y%m%d-%H%M%S")}')

    predictPosesVideo(keypointDetect, centerDetect, video_path, output_dir,
                frame_start, number_frames, make_video, skeleton_preset,
                progressBar)

    del centerDetect
    del keypointDetect


def predict3D(project_name, recording_path, weights_center_detect,
            weights_hybridnet, frame_start, number_frames,
            make_videos, skeleton_preset, dataset_name, progressBar = None):
    project = ProjectManager()
    if not project.load(project_name):
        return
    hybridNet = HybridNet('inference', project.cfg, weights_hybridnet)
    centerDetect = EfficientTrack('CenterDetectInference', project.cfg,
                weights_center_detect)

    output_dir = os.path.join(project.parent_dir,
                project.cfg.PROJECTS_ROOT_PATH, project_name,
                'predictions', f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}')

    reproTools = load_reprojection_tools(project.cfg)
    if len(reproTools) == 1:
        reproTool = reproTools[list(reproTools.keys())[0]]
    elif len(reproTools) > 1:
        if dataset_name == None:
            reproTool = reproTools[list(reproTools.keys())[0]]
        else:
            reproTool = reproTools[dataset_name]
    else:
        print (f'{CLIColors.FAIL}Could not load reprojection Tool for specified '
                    f'project...{CLIColors.ENDC}')
        return
    predictPosesVideos(hybridNet, centerDetect, reproTool, recording_path,
                output_dir, frame_start, number_frames, make_videos,
                   skeleton_preset, progressBar)
    del centerDetect
    del hybridNet
