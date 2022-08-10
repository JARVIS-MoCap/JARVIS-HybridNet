"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import os
import torch
import streamlit as st
from contextlib import ExitStack

from jarvis.config.project_manager import ProjectManager
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.hybridnet.hybridnet import HybridNet
import jarvis.utils.clp as clp
from jarvis.utils.utils import get_available_pretrains


def load_weights_keypoint_detect(model, weights_path = None):
    if weights_path is not None:
        if os.path.isfile(weights_path):
            pretrained_dict = torch.load(weights_path)
            model.load_state_dict(pretrained_dict, strict=False)
            clp.info(f'Successfully loaded weights: {weights_path}')
            return True
        else:
            return False
    else:
        return True


def get_latest_weights_efficienttrack(cfg, mode):
    search_path = os.path.join(cfg.PARENT_DIR, 'projects',
                               cfg.PROJECT_NAME, 'models', mode)
    dirs = os.listdir(search_path)
    dirs = [os.path.join(search_path, d) for d in dirs]
    dirs.sort(key=lambda x: os.path.getmtime(x))
    dirs.reverse()
    for weights_dir in dirs:
        weigths_path = os.path.join(weights_dir,
                    f'EfficientTrack-{cfg.KEYPOINTDETECT.MODEL_SIZE}'
                    f'_final.pth')
        if os.path.isfile(weigths_path):
            return weigths_path
    return None



def train_efficienttrack(mode, project_name, num_epochs, weights,
            streamlitWidgets = None, **kwargs):
    torch.backends.cudnn.benchmark = True
    camera_list = None
    run_name = None
    if 'cameras_to_use' in kwargs:
        camera_list = kwargs['cameras_to_use']
    if 'run_name' in kwargs:
        run_name = kwargs['run_name']
    with ExitStack() as stack:
        if streamlitWidgets != None:
            gs = stack.enter_context(st.spinner('Preparing Model for '
                        'training...'))
        project = ProjectManager()
        if  not project.load(project_name):
            return
        project_path = ''
        if os.path.isabs(project_name):
            project_path = os.path.join(project_name, 'logs', mode)
        else:
            project_path = os.path.join(project.cfg.PARENT_DIR,
                        project.cfg.PROJECTS_ROOT_PATH, project_name,
                        'logs', mode)
        if num_epochs == None:
            if (mode == 'CenterDetect'):
                num_epochs = project.cfg.CENTERDETECT.NUM_EPOCHS
            if (mode == 'KeypointDetect'):
                num_epochs = project.cfg.KEYPOINTDETECT.NUM_EPOCHS
        clp.info(f'Training {mode} on project {project_name} for '
                    f'{num_epochs} epochs!')

        training_set = Dataset2D(project.cfg, set='train', mode = mode,
                    cameras_to_use = camera_list)
        val_set = Dataset2D(project.cfg, set='val',mode = mode,
                    cameras_to_use = camera_list)
        efficientTrack = EfficientTrack(mode, project.cfg,
                    run_name = run_name)


        pose_pretrain_list = get_available_pretrains(project.cfg.PARENT_DIR)
        if weights == "latest":
            weights = efficientTrack.get_latest_weights()
            if weights == None:
                clp.warning('Could not find previously saved weights, '
                       f'using random initialization instead')
            found_weights = efficientTrack.load_weights(weights)
        elif weights == "None" or weights == None:
            found_weights = True
        elif weights == "ecoset" or weights == "EcoSet":
            found_weights = efficientTrack.load_ecoset_pretrain()
        elif weights in pose_pretrain_list:
            found_weights = efficientTrack.load_pose_pretrain(weights)
        else:
            found_weights = efficientTrack.load_weights(weights)
        if not found_weights:
            clp.error('Could not load weights from specified '
                        f'path...')
            return False

    train_results = efficientTrack.train(training_set, val_set, num_epochs,
                streamlitWidgets = streamlitWidgets)
    clp.success('Succesfully finished training!')
    print ('Final Stats:')
    print (f'Training Loss: {train_results["train_loss"]}')
    print (f'Training Accuracy [px]: {train_results["train_acc"]}')
    print (f'Validation Loss: {train_results["val_loss"]}')
    print (f'Validation Accuracy [px]: {train_results["val_acc"]}')
    print ()
    del efficientTrack
    return True


def train_hybridnet(project_name, num_epochs, weights_keypoint_detect, weights,
            mode, finetune = False, streamlitWidgets = None, **kwargs):
    camera_list = None
    run_name = None
    if 'cameras_to_use' in kwargs:
        camera_list = kwargs['cameras_to_use']
    if 'run_name' in kwargs:
        run_name = kwargs['run_name']
    with ExitStack() as stack:
        if streamlitWidgets != None:
            gs = stack.enter_context(st.spinner('Preparing Model for '
                        'training...'))
        project = ProjectManager()
        if  not project.load(project_name):
            return
        project_path = ''
        if os.path.isabs(project_name):
            project_path = os.path.join(project_name, 'logs', 'HybridNet')
        else:
            project_path = os.path.join(project.cfg.PARENT_DIR,
                        project.cfg.PROJECTS_ROOT_PATH, project_name, 'logs',
                        'HybridNet')
        if num_epochs == None:
            num_epochs = project.cfg.HYBRIDNET.NUM_EPOCHS
        clp.info(f'Training HybridNet on project {project_name} for {num_epochs}'
                    f' epochs!')

        training_set = Dataset3D(project.cfg, set='train',
                    cameras_to_use = camera_list)
        val_set = Dataset3D(project.cfg, set='val',
                    cameras_to_use = camera_list)
        hybridNet = HybridNet('train', project.cfg,
                    run_name = run_name)

        effTrack = hybridNet.model.effTrack

        pose_pretrain_list = get_available_pretrains(project.cfg.PARENT_DIR)
        if weights in pose_pretrain_list:
            found_weights = hybridNet.load_pose_pretrain(weights)

        else:
            if weights_keypoint_detect == "latest":
                weights_keypoint_detect = get_latest_weights_efficienttrack(
                            project.cfg, 'KeypointDetect')
                if weights_keypoint_detect == None:
                    clp.warning('Could not find previously saved weights'
                           ' for KeypointDetect, using random initialization '
                           'instead')
                found_weights = load_weights_keypoint_detect(effTrack,
                            weights_keypoint_detect)
            elif (weights_keypoint_detect == "None"
                        or weights_keypoint_detect == None):
                found_weights = True
            else:
                found_weights = load_weights_keypoint_detect(effTrack,
                            weights_keypoint_detect)
            if not found_weights:
                clp.error('{Could not load weights from specified '
                            f'path...')
                return

            if weights == "latest":
                weights = hybridNet.get_latest_weights()
                if weights == None:
                    clp.warning('Could not find previously saved weights, '
                           'using random initialization instead')
                found_weights = hybridNet.load_weights(weights)
            elif weights == "None" or weights == None:
                found_weights = True
            else:
                found_weights = hybridNet.load_weights(weights)
        if not found_weights:
            clp.error('{Could not load weights from specified '
                        f'path...')
            return

    hybridNet.set_training_mode(mode)
    if finetune:
        hybridNet.cfg.HYBRIDNET.MAX_LEARNING_RATE = \
                    hybridNet.cfg.HYBRIDNET.MAX_LEARNING_RATE/10
    train_results = hybridNet.train(training_set, val_set, num_epochs,
                streamlitWidgets = streamlitWidgets)
    clp.success('Succesfully finished training!')
    print ('Final Stats:')
    print (f'Training Loss: {train_results["train_loss"]}')
    print (f'Training Accuracy [mm]: {train_results["train_acc"]}')
    print (f'Validation Loss: {train_results["val_loss"]}')
    print (f'Validation Accuracy [mm]: {train_results["val_acc"]}')
    print ()
    del hybridNet
