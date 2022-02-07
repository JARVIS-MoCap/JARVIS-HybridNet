import os
import streamlit as st
from streamlit_option_menu import option_menu
import jarvis.config.project_manager as ProjectManager
import jarvis.train_interface as train
import jarvis.predict_interface as predict
import jarvis.visualize_interface as visualize
import time


def predict2D_gui(project):
    st.header("Predict 2D")
    st.write("Predict the 2D Keypoints for a single Video.")
    with st.form("predict_2D_form"):
        video_path = st.text_input("Path of Video:",
                    placeholder = "Please enter path...")
        output_dir = st.text_input("Output Directory:",
                    placeholder = "Please enter path...")
        col3, col4 = st.columns(2)
        with col3:
            weights_center_detect = st.text_input("Weights for CenterDetect:",
                        value = 'latest')
        with col4:
            weights_keypoint_detect = st.text_input("Weights for KeypointDetect:",
                        value = 'latest')
        skeleton_preset = st.selectbox('Skeleton Preset',
                    ['None', 'Hand', 'HumanBody', 'RodentBody'])
        make_video = st.checkbox("Make Video overlayed with predictions?",
                    value = True)
        col1, col2 = st.columns(2)
        with col1:
            frame_start = st.number_input("Start Frame:",
                        value = 0, min_value = 0)
        with col2:
            number_frames = st.number_input("Number of Frames:",
                        value = -1, min_value = -1)
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not os.path.isfile(video_path):
            st.error("Video File does not exist!")
            return
        if output_dir == "":
            st.error("Please enter a valid ouput directory!")
            return
        if skeletonPreset == "None":
            skeletonPreset = None
        st.subheader("Prediction Progress:")
        my_bar = st.progress(0)
        predict.predict2D(project, video_path, weights_center_detect,
                    weights_keypoint_detect, output_dir, frame_start,
                    number_frames, make_video, skeleton_preset, my_bar)
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()


def predict3D_gui(project):
    st.header("Predict 3D")
    st.write("Predict the 3D Keypoints for a set of synchronously recorded videos.")
    with st.form("predict_3D_form"):
        recording_path = st.text_input("Path of recording directory:",
                    placeholder = "Please enter path...")
        output_dir = st.text_input("Output Directory:",
                    placeholder = "Please enter path...")
        col3, col4 = st.columns(2)
        with col3:
            weights_center_detect = st.text_input("Weights for CenterDetect:",
                        value = "latest")
        with col4:
            weights_hybridnet = st.text_input("Weights for HybridNet:",
                        value = "latest")
        skeleton_preset = st.selectbox('Skeleton Preset',
                    ['None', 'Hand', 'HumanBody', 'RodentBody'])
        make_videos = st.checkbox("Make Videos overlayed with predictions?",
                    value = True)
        col1, col2 = st.columns(2)
        with col1:
            frame_start = st.number_input("Start Frame:",
                        value = 0, min_value = 0)
            with col2:
                number_frames = st.number_input("Number of Frames:",
                            value = -1, min_value = -1)
        submitted = st.form_submit_button("Predict")
    if submitted:
        if not os.path.isdir(recording_path):
            st.error("Recording directory does not exist!")
            return
        if output_dir == "":
            st.error("Please enter a valid ouput directory!")
            return
        if skeleton_preset == "None":
            skeleton_preset = None
        st.subheader("Prediction Progress:")
        my_bar = st.progress(0)
        #TODO: make this work when there aremultiple calibration sets in a dataset
        predict.predict3D(project, recording_path, weights_center_detect,
                    weights_hybridnet, output_dir, frame_start, number_frames,
                    make_videos, skeleton_preset, None, my_bar)
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()
