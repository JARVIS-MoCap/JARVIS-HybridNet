"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import os
import time
import streamlit as st
from streamlit_option_menu import option_menu

import jarvis.config.project_manager as ProjectManager
import jarvis.train_interface as train
from jarvis.prediction.predict2D import predict2D
from jarvis.prediction.predict3D import predict3D
from jarvis.utils.paramClasses import Predict2DParams, Predict3DParams


def predict2D_gui(project):
    st.header("Predict 2D")
    st.write("Predict the 2D Keypoints for a single Video.")
    with st.form("predict_2D_form"):
        video_path = st.text_input("Path of Video:",
                    placeholder = "Please enter path...")

        params = Predict2DParams(project, video_path)
        col3, col4 = st.columns(2)
        with col3:
            params.weights_center_detect = st.text_input(
                        "Weights for CenterDetect:",
                        value = 'latest',
                        help = "Use 'latest' to load you last saved weights, "
                        "or specify the path to a '.pth' file.")
        with col4:
            params.weights_keypoint_detect = st.text_input(
                        "Weights for KeypointDetect:",
                        value = 'latest',
                        help = "Use 'latest' to load you last saved weights, "
                        "or specify the path to a '.pth' file.")
        col1, col2 = st.columns(2)
        with col1:
            params.frame_start = st.number_input("Start Frame:",
                        value = 0, min_value = 0)
        with col2:
            params.number_frames = st.number_input("Number of Frames:",
                        value = -1, min_value = -1)
        submitted = st.form_submit_button("Predict")
    if submitted:
        projectManager = ProjectManager.ProjectManager()
        projectManager.load(project)
        cfg = projectManager.cfg
        if not os.path.isfile(video_path):
            st.error("Video File does not exist!")
            return
        if not (params.weights_center_detect == "latest"
                    or (os.path.isfile(params.weights_center_detect)
                    and params.weights_center_detect.split(".")[-1] == "pth")):
            st.error("CenterDetect weights do not exist!")
            return
        if not (params.weights_keypoint_detect == "latest"
                    or (os.path.isfile(params.weights_keypoint_detect)
                    and params.weights_center_detect.split(".")[-1] == "pth")):
            st.error("KeypointDetect weights do not exist!")
            return
        st.subheader("Prediction Progress:")
        params.progress_bar = st.progress(0)
        predict2D(params)
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()


def predict3D_gui(project):
    st.header("Predict 3D")
    st.write("Predict the 3D Keypoints for a set of synchronously recorded "
                "videos.")
    with st.form("predict_3D_form"):
        recording_path = st.text_input("Path of recording directory:",
                    placeholder = "Please enter path...")
        params = Predict3DParams(project, recording_path)
        col3, col4 = st.columns(2)
        with col3:
            params.weights_center_detect = st.text_input(
                        "Weights for CenterDetect:",
                        value = "latest",
                        help = "Use 'latest' to load you last saved weights, "
                        "or specify the path to a '.pth' file.")
        with col4:
            params.weights_hybridnet = st.text_input("Weights for HybridNet:",
                        value = "latest",
                        help = "Use 'latest' to load you last saved weights, "
                        "or specify the path to a '.pth' file.")
        col1, col2 = st.columns(2)
        with col1:
            params.frame_start = st.number_input("Start Frame:",
                        value = 0, min_value = 0)
        with col2:
            params.number_frames = st.number_input("Number of Frames:",
                        value = -1, min_value = -1)
        projectManager = ProjectManager.ProjectManager()
        projectManager.load(project)
        cfg = projectManager.cfg
        dataset_name = cfg.DATASET.DATASET_3D
        if os.path.isabs(dataset_name):
            calib_root_path = os.path.join(dataset_name, 'calib_params')
        else:
            calib_root_path = os.path.join(cfg.PARENT_DIR,
                        cfg.DATASET.DATASET_ROOT_DIR, dataset_name,
                        'calib_params')
        if os.path.isdir(calib_root_path):
            calibrations = os.listdir(calib_root_path)
        else:
            calibrations = []
        if len(calibrations) != 1:
            calibration_selection = st.selectbox('Select the CalibrationSet '
                        'you want to use', calibrations)
        else:
            calibration_selection = None
        params.dataset_name = calibration_selection
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not os.path.isdir(recording_path):
            st.error("Recording directory does not exist!")
            return
        if not (params.weights_center_detect == "latest"
                    or (os.path.isfile(params.weights_center_detect)
                    and params.weights_center_detect.split(".")[-1] == "pth")):
            st.error("CenterDetect weights do not exist!")
            return
        if not (params.weights_hybridnet == "latest"
                    or (os.path.isfile(params.weights_hybridnet)
                    and params.weights_hybridnet.split(".")[-1] == "pth")):
            st.error("HybridNet weights do not exist!")
            return
        st.subheader("Prediction Progress:")
        params.progress_bar = st.progress(0)
        predict3D(params)
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()
