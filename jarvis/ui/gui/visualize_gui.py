import os
import streamlit as st
from streamlit_option_menu import option_menu

import jarvis.config.project_manager as ProjectManager
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
import jarvis.visualize_interface as vis_interface


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
    if mode == 'KeypointDetect':
        skeleton_preset = st.selectbox('Skeleton Preset',
                    ['None', 'Hand', 'HumanBody', 'MonkeyBody', 'RodentBody'])
    else:
        skeleton_preset = None
    img = vis_interface.visualize_2D_sample(set, mode, img_idx, skeleton_preset)
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        st.write("")
    with col2:
        st.image(img, use_column_width = 'always')
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
    skeleton_preset = st.selectbox('Skeleton Preset',
                ['None', 'Hand', 'HumanBody', 'MonkeyBody', 'RodentBody'])
    set = Dataset3D(projectManager.cfg, set=set_name)
    img_idx = st.slider("Frame Index", min_value=0, max_value=len(set.image_ids), value=1)
    azim = st.slider("Azim", min_value=0, max_value=180)
    elev = st.slider("Elev", min_value=0, max_value=180)

    fig = vis_interface.visualize_3D_sample(set, img_idx, skeleton_preset, azim, elev)
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        st.write("")
    with col2:
        st.pyplot(fig)
    with col3:
        st.write("")
