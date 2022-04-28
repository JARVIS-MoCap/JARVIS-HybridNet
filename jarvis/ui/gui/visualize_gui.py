import os
import streamlit as st
from ruamel.yaml import YAML
from streamlit_option_menu import option_menu

import jarvis.config.project_manager as ProjectManager
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
import jarvis.visualization.visualize_dataset as visualize_dataset
from jarvis.visualization.create_videos3D import create_videos3D
from jarvis.visualization.create_videos2D import create_videos2D
from jarvis.utils.paramClasses import CreateVideos3DParams, CreateVideos2DParams
import time


def create_video2D_gui(project):
    st.header("Create Video 2D")
    st.write("Create annotated Video from 2D Predictions")

    projectManager = ProjectManager.ProjectManager()
    projectManager.load(project)
    cfg = projectManager.cfg
    prediction_options, predict_path = get_prediction_paths(cfg, '2D')
    if prediction_options == None:
        st.warning("No predictions created yet. Please run Predict2D first!")
        return
    prediction = st.selectbox('Select Prediction to load',
                prediction_options)
    prediction_path = os.path.join(predict_path, prediction)
    if prediction_path == None:
        st.error("Please make sure you created valid "
                    "Predicions with 'Predict2D!'")
    data_csvs = get_data_csv(prediction_path)
    data_csv_name = st.selectbox("Select Prediction '.csv' to use",
                data_csvs)
    data_csv = os.path.join(prediction_path, data_csv_name)

    if st.button("Create Video"):

        with open(os.path.join(prediction_path, 'info.yaml')) as file:
            yaml = YAML()
            info_yaml = yaml.load(file)
            recording_path = info_yaml['recording_path']
            frame_start = info_yaml['frame_start']
            number_frames = info_yaml['number_frames']

        params = CreateVideos2DParams(project, recording_path, data_csv)

        params.frame_start = frame_start
        params.number_frames = number_frames

        params.progress_bar = st.progress(0)
        create_videos2D(params)
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()


def create_video3D_gui(project):
    st.header("Create Video 3D")
    st.write("Create annotated Video from 3D Predictions")

    projectManager = ProjectManager.ProjectManager()
    projectManager.load(project)
    cfg = projectManager.cfg
    prediction_options, predict_path = get_prediction_paths(cfg, '3D')
    if prediction_options == None:
        st.warning("No predictions created yet. Please run Predict3D first!")
        return
    prediction = st.selectbox('Select Prediction to load',
                prediction_options)
    prediction_path = os.path.join(predict_path, prediction)
    if prediction_path == None:
        st.error("Please make sure you created valid "
                    "Predicions with 'Predict3D!'")
    data_csvs = get_data_csv(prediction_path)
    data_csv_name = st.selectbox("Select Prediction '.csv' to use",
                data_csvs)
    data_csv = os.path.join(prediction_path, data_csv_name)

    with open(os.path.join(prediction_path, 'info.yaml')) as file:
        yaml = YAML()
        info_yaml = yaml.load(file)
        recording_path = info_yaml['recording_path']
        frame_start = info_yaml['frame_start']
        number_frames = info_yaml['number_frames']

    cameras = []
    videos = os.listdir(recording_path)
    for video in os.listdir(recording_path):
        cameras.append(video.split('.')[0])

    video_cam_list = st.multiselect("Select Cameras to create Videos for",
                options = cameras, default = cameras)

    if st.button("Create Video"):
        params = CreateVideos3DParams(project, recording_path, data_csv)
        print ("CSV:", params.data_csv)

        params.video_cam_list = video_cam_list
        params.frame_start = frame_start
        params.number_frames = number_frames

        params.progress_bar = st.progress(0)
        create_videos3D(params)
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()


def visualize2D_gui(project):
    st.header("Visualize Dataset 2D")
    projectManager = ProjectManager.ProjectManager()
    projectManager.load(project)
    set_name = st.selectbox(
                'Select Dataset part to be visualized', ['val', 'train'])
    mode = st.selectbox(
                'Select Mode', ['CenterDetect', 'KeypointDetect'])
    set = Dataset2D(projectManager.cfg, set=set_name, mode = mode)
    img_idx = st.slider("Frame Index", min_value=0, max_value=len(set.image_ids), value=1)
    fig = visualize_dataset.visualize_2D_sample(set, mode, img_idx)
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        st.write("")
    with col2:
        st.pyplot(fig)
    with col3:
        st.write("")


def visualize3D_gui(project):
    st.header("Visualize Dataset 3D")
    projectManager = ProjectManager.ProjectManager()
    projectManager.load(project)
    set_name = st.selectbox(
                'Select Dataset part to be visualized', ['val', 'train'])
    mode = st.selectbox(
                'Select Mode', ['CenterDetect', 'KeypointDetect'])
    set = Dataset3D(projectManager.cfg, set=set_name)
    img_idx = st.slider("Frame Index", min_value=0, max_value=len(set.image_ids), value=1)
    azim = st.slider("Azim", min_value=0, max_value=180)
    elev = st.slider("Elev", min_value=0, max_value=180)

    fig = visualize_dataset.visualize_3D_sample(set, img_idx, azim, elev)
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        st.write("")
    with col2:
        st.pyplot(fig)
    with col3:
        st.write("")


def get_prediction_paths(cfg, mode):
    predict_path = os.path.join(cfg.PARENT_DIR,
                cfg.PROJECTS_ROOT_PATH, cfg.PROJECT_NAME,
                'predictions', f'predictions{mode}')

    if (not os.path.exists(predict_path)) or len(os.listdir(predict_path)) == 0:
        return None, None

    prediction_options = sorted(os.listdir(predict_path))[::-1]
    return prediction_options, predict_path

def get_data_csv(path):
    files = os.listdir(path)
    files = [file for file in files if file.split(".")[-1] == 'csv']
    return files
