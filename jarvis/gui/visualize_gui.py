import os
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib
import jarvis.config.project_manager as ProjectManager
import jarvis.visualize_interface as visualize
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
import jarvis.prediction.prediction_utils as utils
import jarvis.visualization.visualization_utils as viz_utils


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.4*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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
	sample = set.__getitem__(img_idx)
	img = (sample[0]*projectManager.cfg.DATASET.STD+projectManager.cfg.DATASET.MEAN)*255
	heatmaps = sample[1]
	keypoints = sample[2]
	img = img - np.min(img)
	img = img/np.max(img)*255
	img = cv2.resize(img, None, fx=2, fy = 2)
	if mode == 'CenterDetect':
		if keypoints[0,0] + keypoints[0,1] != 0:
			img = cv2.circle(img, (int(keypoints[0,0]*2),int(keypoints[0,1]*2)), 4, (255,0,0), 6)
	else:
		skeletonPreset = st.selectbox('Skeleton Preset',
					['None', 'Hand', 'HumanBody', 'RodentBody'])
		colors = []
		line_idxs = []
		if skeletonPreset != "None":
	 		colors, line_idxs = utils.get_colors_and_lines(skeletonPreset)
		else:
			cmap = matplotlib.cm.get_cmap('jet')
			for i in range(projectManager.cfg.KEYPOINTDETECT.NUM_JOINTS):
				colors.append(((np.array(
				cmap(float(i)/projectManager.cfg.KEYPOINTDETECT.NUM_JOINTS)) *
							255).astype(int)[:3]).tolist())
		keypoints = keypoints.reshape(-1,3)
		for i,keypoint in enumerate(keypoints):
			if keypoint[0] + keypoint[1] != 0:
				img = cv2.circle(img, (int(keypoint[0]*2),int(keypoint[1]*2)), 4, colors[i], 6)
		for line in line_idxs:
			if keypoints[line[0]][0] + keypoints[line[0]][1] != 0 and keypoints[line[1]][0] + keypoints[line[1]][1] != 0:
				cv2.line(img,
					(int(keypoints[line[0]][0]*2), int(keypoints[line[0]][1]*2)),
					(int(keypoints[line[1]][0]*2), int(keypoints[line[1]][1]*2)),
					colors[line[1]], 1)

	img = img/255
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
	skeletonPreset = st.selectbox('Skeleton Preset',
				['None', 'Hand', 'HumanBody', 'RodentBody'])
	colors = []
	line_idxs = []
	if skeletonPreset != "None":
			colors, line_idxs = viz_utils.get_colors_and_lines(skeletonPreset)
	else:
		cmap = matplotlib.cm.get_cmap('jet')
		for i in range(projectManager.cfg.KEYPOINTDETECT.NUM_JOINTS):
			colors.append(((np.array(
			cmap(float(i)/projectManager.cfg.KEYPOINTDETECT.NUM_JOINTS)) *
						255).astype(int)[:3]).tolist())
	set = Dataset3D(projectManager.cfg, set=set_name, mode = mode)
	img_idx = st.slider("Frame Index", min_value=0, max_value=len(set.image_ids), value=1)
	sample = set.__getitem__(img_idx)
	images = sample[0]
	keypoints3D = sample[1]

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_axis_off()
	ax.margins(0)
	azim = st.slider("Azim", min_value=0, max_value=180)
	elev = st.slider("Elev", min_value=0, max_value=180)

	ax.azim = azim
	ax.elev = elev
	for i, point in enumerate(keypoints3D):
		ax.scatter(point[0], point[1], point[2], color = tuple(np.array(colors[i])/255.))
	for line in line_idxs:
		ax.plot([keypoints3D[line[0]][0], keypoints3D[line[1]][0]],
				  [keypoints3D[line[0]][1], keypoints3D[line[1]][1]],
				  [keypoints3D[line[0]][2], keypoints3D[line[1]][2]],
				  color = tuple(np.array(colors[line[1]])/255.))
	set_axes_equal(ax)
	ax.autoscale_view('tight')
	col1, col2, col3 = st.columns([1,3,1])
	with col1:
		st.write("")
	with col2:
		st.pyplot(fig)
	with col3:
		st.write("")
