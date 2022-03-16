import os
import torch
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
    predictPosesVideo(keypointDetect, centerDetect, video_path,
                frameStart = frame_start, numberFrames = number_frames,
                make_video = make_video, skeletonPreset = skeleton_preset,
                progressBar = progressBar)
    del centerDetect
    del keypointDetect


def predict3D(project_name, recording_path, weights_center_detect,
            weights_hybridnet, output_dir, frame_start, number_frames,
            make_videos, skeleton_preset, dataset_name, progressBar = None):
    project = ProjectManager()
    if not project.load(project_name):
        return
    hybridNet = HybridNet('inference', project.cfg, weights_hybridnet)
    centerDetect = EfficientTrack('CenterDetectInference', project.cfg,
                weights_center_detect)
    if output_dir == None:
        output_dir = 'PosePredictions'

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
    predictPosesVideos(hybridNet,
                   centerDetect,
                   reproTool,
                   recording_path = recording_path,
                   output_dir = output_dir,
                   frameStart = frame_start,
                   numberFrames = number_frames,
                   make_videos = make_videos,
                   skeletonPreset = skeleton_preset,
                   progressBar = progressBar)
    del centerDetect
    del hybridNet
