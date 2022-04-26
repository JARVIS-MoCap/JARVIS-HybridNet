import os
import streamlit as st
from streamlit_option_menu import option_menu
import jarvis.analyze_interface as analyze_interface
from jarvis.config.project_manager import ProjectManager



def analyze_validation_set_gui(project_name):
    with st.form("analysis_form"):
        projectManager = ProjectManager()
        projectManager.load(project_name)
        cfg = projectManager.get_cfg()
        col3, col4 = st.columns(2)
        with col3:
            weights_center = st.text_input("Weights for CenterDetect:",
                        value = 'latest',
                        help = "Use 'latest' to load you last saved weights, or "
                                    "specify the path to a '.pth' file.")
        with col4:
            weights_hybridnet = st.text_input("Weights for HybridNet:",
                        value = 'latest',
                        help = "Use 'latest' to load you last saved weights, or "
                                    "specify the path to a '.pth' file.")
        dataset_name = cfg.DATASET.DATASET_3D
        if os.path.isabs(dataset_name):
            calib_root_path = os.path.join(dataset_name, 'calib_params')
        else:
            calib_root_path = os.path.join(cfg.PARENT_DIR,
                        cfg.DATASET.DATASET_ROOT_DIR, dataset_name,
                        'calib_params')
        calib_path = os.path.join(calib_root_path,os.listdir(calib_root_path)[0])
        camera_names = os.listdir(calib_path)
        camera_names = [cam.split(".")[0] for cam in camera_names]
        cameras_to_use = st.multiselect("Select Cameras to use for analysis",
                    options = camera_names, default = camera_names)
        submitted = st.form_submit_button("Analyse")
    if submitted:
        progress_bar = st.progress(0)
        analyze_interface.analyze_validation_data(project_name, weights_center,
                    weights_hybridnet, cameras_to_use, progress_bar = progress_bar)

def plot_error_histogram_gui(project_name):
    return

def plot_errors_per_keypoint_gui(project_name):
    return
