"""
jarvis_gui.py
============
Main Streamlit GUI script.
"""

import streamlit as st
import torch
import os
from streamlit_option_menu import option_menu
import streamlit.config
import json, yaml

import jarvis.config.project_manager as ProjectManager
import jarvis.train_interface as train
import jarvis.visualize_interface as visualize
from jarvis.gui.predict_gui import predict2D_gui, predict3D_gui
from jarvis.gui.train_gui import train_all_gui, train_center_detect_gui, \
		train_keypoint_detect_gui, train_hybridnet_gui


st.set_page_config(
	layout="wide",
	initial_sidebar_state="auto",
	page_title="JARVIS Dashboard",
	page_icon=None,
)

st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-color: #64a420;
        }
    </style>""",
    unsafe_allow_html=True,
)


def create_project_clicked(project_name, dataset3D, dataset2D):
    if project_name != "" and (dataset3D != "" or dataset2D != ""):
        if dataset2D == "":
            dataset2D = dataset3D
        if dataset3D == "":
            dataset3D = None
        projectManager.get_create_config_interactive(project_name,
					dataset2D, dataset3D)
    else:
        st.warning('Missing Project Name or at least one dataset path')


projectManager = ProjectManager.ProjectManager()

projects = projectManager.get_projects()
projects = ['Select...'] + projects#



if 'created_project' in st.session_state:
    default_idx = projects.index(st.session_state['created_project'])
    del st.session_state['created_project']

else:
	default_idx = 0

st.sidebar.title("Project:")
project_box = st.sidebar.selectbox(
'Select Project to be loaded', projects, index = default_idx,
help = "Load existing project, or choose 'Select...' to create new one")


if 'results_available' in st.session_state:
	del st.session_state['results_available']
	st.title("Training Results")

	if  'CenterDetect/Train Loss' in st.session_state:
		train_losses = st.session_state['CenterDetect/Train Loss']
		val_losses = st.session_state['CenterDetect/Val Loss']
		val_accs = st.session_state['CenterDetect/Val Accuracy']
		display_results = True

		del st.session_state['CenterDetect/Train Loss']
		del st.session_state['CenterDetect/Val Loss']
		del st.session_state['CenterDetect/Val Accuracy']
		st.header("CenterDetect")
		with st.expander("Expand CenterDetect Results", expanded = True):
			st.subheader('Loss')
			st.line_chart({'Train Loss': train_losses, 'Val Loss': val_losses})
			st.subheader('Accuracy')
			st.line_chart({'Val Accuracy [px]': val_accs})

	if  'KeypointDetect/Train Loss' in st.session_state:
		train_losses = st.session_state['KeypointDetect/Train Loss']
		val_losses = st.session_state['KeypointDetect/Val Loss']
		val_accs = st.session_state['KeypointDetect/Val Accuracy']
		display_results = True

		del st.session_state['KeypointDetect/Train Loss']
		del st.session_state['KeypointDetect/Val Loss']
		del st.session_state['KeypointDetect/Val Accuracy']
		st.header("KeypointDetect")
		with st.expander("Expand KeypointDetect Results", expanded = True):
			st.subheader('Loss')
			st.line_chart({'Train Loss': train_losses, 'Val Loss': val_losses})
			st.subheader('Accuracy')
			st.line_chart({'Val Accuracy [px]': val_accs})

	if 'HybridNet/3D_only/Train Loss' in st.session_state:
		train_losses = st.session_state['HybridNet/3D_only/Train Loss']
		val_losses = st.session_state['HybridNet/3D_only/Val Loss']
		train_accs = st.session_state['HybridNet/3D_only/Train Accuracy']
		val_accs = st.session_state['HybridNet/3D_only/Val Accuracy']
		del st.session_state['HybridNet/3D_only/Train Loss']
		del st.session_state['HybridNet/3D_only/Val Loss']
		del st.session_state['HybridNet/3D_only/Train Accuracy']
		del st.session_state['HybridNet/3D_only/Val Accuracy']
		st.header("HybridNet")
		with st.expander("Expand HybridNet Results", expanded = True):
			st.subheader('Loss')
			st.line_chart({'Train Loss': train_losses, 'Val Loss': val_losses})
			st.subheader('Accuracy')
			st.line_chart({'Train Accuracy [mm]': train_accs, 'Val Accuracy [mm]': val_accs})

	if 'HybridNet/all/Train Loss' in st.session_state:
		train_losses = st.session_state['HybridNet/all/Train Loss']
		val_losses = st.session_state['HybridNet/all/Val Loss']
		train_accs = st.session_state['HybridNet/all/Train Accuracy']
		val_accs = st.session_state['HybridNet/all/Val Accuracy']
		del st.session_state['HybridNet/all/Train Loss']
		del st.session_state['HybridNet/all/Val Loss']
		del st.session_state['HybridNet/all/Train Accuracy']
		del st.session_state['HybridNet/all/Val Accuracy']
		st.header("HybridNet Finetuning")
		with st.expander("Expand HybridNet Finetuning Results", expanded = True):
			st.subheader('Loss')
			st.line_chart({'Train Loss': train_losses, 'Val Loss': val_losses})
			st.subheader('Accuracy')
			st.line_chart({'Train Accuracy [mm]': train_accs, 'Val Accuracy [mm]': val_accs})

	st.button("Clear")

elif project_box != 'Select...':
	projectManager.load(project_box)
	with st.sidebar:
		selected = option_menu("Menus", ['Training', 'Prediction', 'Visualization'],
		icons=['gpu-card', 'layer-forward', 'graph-up'],
		menu_icon="list", default_index=0)

	if selected == "Training":
		with st.sidebar:
			train_mode = option_menu("Training Options",
						['Train Full', 'Train Center', 'Train Keypoint', 'Train HybridNet'],
						menu_icon="gpu-card", default_index=0)
		st.title("Training")
		if train_mode == 'Train Full':
			train_all_gui(project_box, projectManager.cfg)
		elif train_mode == 'Train Center':
			train_center_detect_gui(project_box, projectManager.cfg)
		elif train_mode == "Train Keypoint":
			train_keypoint_detect_gui(project_box, projectManager.cfg)
		else:
			train_hybridnet_gui(project_box, projectManager.cfg)

	elif selected == "Prediction":
		with st.sidebar:
			predict_mode = option_menu("Prediction Options", ['Predict3D', 'Predict2D'],
				icons=['badge-3d', 'camera-video'],
				menu_icon="layer-forward", default_index=0)
		st.title("Prediction")
		if predict_mode == 'Predict3D':
			predict3D_gui(project_box)
		else:
			predict2D_gui(project_box)
	else:
		with st.sidebar:
			vis_mode = option_menu("Visualizations", ['Test', 'Test2'],
						menu_icon="graph-up", default_index=0)
		st.title("Coming Soon :)")

	with st.container():
		st.header('Project Info')
		f = open(os.path.join(projectManager.parent_dir, 'projects',
					project_box,'config.yaml'))
		with st.expander("Project Config"):
			st.json(json.dumps(yaml.safe_load(f)))

else:
    st.title("Create Project:")
    with st.form("create_project_form"):
        project_name = st.text_input("Project Name:",
					placeholder = "Please enter Name...")
        dataset3D = st.text_input("Dataset3D:",
					placeholder = "Please enter path to 3D dataset...",
					help = 'Make sure 3D Dataset was selected when creating '
						   'the dataset specified here')
        dataset2D = st.text_input("Dataset2D:",
					placeholder = "Please enter path to 2D dataset...",
					help = 'This dataset will be used to train the Center and '
					'2D Keypoint Detectors. Use this if you do not have multi '
					'camera annotations or you want to use a larger '
					'pretraining set.')
        submitted = st.form_submit_button("Create Project")
    if submitted or ('creating_project' in st.session_state
				and st.session_state['creating_project']):
        create_project_clicked(project_name, dataset3D, dataset2D)
