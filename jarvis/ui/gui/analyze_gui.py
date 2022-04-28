import os
import streamlit as st
from streamlit_option_menu import option_menu
import jarvis.analysis.analyze as analyze
import jarvis.analysis.plotting as plotting
from jarvis.config.project_manager import ProjectManager



def analyze_validation_set_gui(project_name):
    st.title("Analyse Validation Data")
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
        analyze.analyze_validation_data(project_name, weights_center,
                    weights_hybridnet, cameras_to_use, progress_bar = progress_bar)


def plot_error_histogram_gui(project_name):
    st.title("Plot Error Histogram")
    path = get_analysis_path(project_name)
    cutoff = st.number_input("Error cutoff for plotting:", -1)
    if (st.button("Plot")):
        fig = plotting.plot_error_histogram(path, cutoff = cutoff,
                    interactive = False)
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            st.write("")
        with col2:
            st.pyplot(fig)
        with col3:
            st.write("")


def plot_errors_per_keypoint_gui(project_name):
    st.title("Plot Error per Keypoint")
    path = get_analysis_path(project_name)
    if (st.button("Plot")):
        fig = plotting.plot_error_per_keypoint(path, project_name,
                    interactive = False)
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            st.write("")
        with col2:
            st.pyplot(fig)
        with col3:
            st.write("")

    return


def get_analysis_path(project_name):
    project = ProjectManager()
    project.load(project_name)
    cfg = project.get_cfg()
    analysis_path = os.path.join(project.parent_dir,
                project.cfg.PROJECTS_ROOT_PATH, project_name,
                'analysis')
    if len(os.listdir(analysis_path)) == 0:
        st.error("Please run Analysis on this project first! Aborting...")
        return None, None
    analysis_set = st.selectbox('Select Prediction to load',
                os.listdir(analysis_path))
    path = os.path.join(analysis_path, analysis_set)
    return path
