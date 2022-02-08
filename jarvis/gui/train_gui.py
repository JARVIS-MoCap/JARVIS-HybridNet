import os
import streamlit as st
from streamlit_option_menu import option_menu
import jarvis.config.project_manager as ProjectManager
import jarvis.train_interface as train
import jarvis.predict_interface as predict
import jarvis.visualize_interface as visualize
import time
import jarvis.train_interface as train

def train_all_gui(project, cfg):
    st.header("Train Full Network")
    st.write("Train all parts of the jarvis network, including CenterDetect and"
                " the 2D and 3D keypoint detectors.")
    with st.form("train_full_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_epochs_center = st.number_input("Epochs CenterDetect:",
                        value = cfg.CENTERDETECT.NUM_EPOCHS,
                        min_value = 1, max_value = 1000)
        with col2:
            num_epochs_keypoint = st.number_input("Epochs KeypointDetect:",
                        value = cfg.KEYPOINTDETECT.NUM_EPOCHS,
                        min_value = 1, max_value = 1000)
        with col3:
            num_epochs_hybridnet = st.number_input("Epochs HybridNet:",
                        value = cfg.HYBRIDNET.NUM_EPOCHS,
                        min_value = 1, max_value = 1000)
        finetune = st.checkbox("Finetune Network", value = True)
        submitted = st.form_submit_button("Train")
    if submitted:
        with st.expander("Expand CenterDetect Training", expanded=True):
            st.header("Training CenterDetect")
            col_center1, col_center2 = st.columns([1,5])
            with col_center1:
                epoch_counter1 = st.empty()
                epoch_counter1.markdown(f"Epoch 0")
            with col_center2:
                progressBar_epoch1 = st.progress(0)
            progressBar_total1 = st.progress(0)
            st.subheader("Loss Monitor")
            plot_loss1 = st.empty()
            st.subheader("Accuracy Monitor")
            plot_acc1 = st.empty()
            train.train_efficienttrack('CenterDetect', project,
                        num_epochs_center, 'ecoset',
                        streamlitWidgets = [progressBar_epoch1, progressBar_total1,
                                            epoch_counter1, plot_loss1, plot_acc1])
        with st.expander("Expand KeypointDetect Training", expanded=True):
            st.header("Training KeypointDetect")
            col_keypoint1, col_keypoint2 = st.columns([1,5])
            with col_keypoint1:
                epoch_counter2 = st.empty()
                epoch_counter2.markdown(f"Epoch 0")
            with col_keypoint2:
                progressBar_epoch2 = st.progress(0)
            progressBar_total2 = st.progress(0)
            st.subheader("Loss Monitor")
            plot_loss2 = st.empty()
            st.subheader("Accuracy Monitor")
            plot_acc2 = st.empty()
            train.train_efficienttrack('KeypointDetect', project,
                        num_epochs_keypoint, 'ecoset',
                        streamlitWidgets = [progressBar_epoch2,progressBar_total2,
                                            epoch_counter2, plot_loss2, plot_acc2])
        with st.expander("Expand HybridNet Training", expanded=True):
            st.header("Training HybridNet")
            col_hybrid1, col_hybrid2 = st.columns([1,5])
            with col_hybrid1:
                epoch_counter3 = st.empty()
                epoch_counter3.markdown(f"Epoch 0")
            with col_hybrid2:
                progressBar_epoch3 = st.progress(0)
            progressBar_total3 = st.progress(0)
            st.subheader("Loss Monitor")
            plot_loss3 = st.empty()
            st.subheader("Accuracy Monitor")
            plot_acc3 = st.empty()
            train.train_hybridnet(project, num_epochs_hybridnet,
                        'latest', None, '3D_only',
                        streamlitWidgets = [progressBar_epoch3,progressBar_total3,
                                            epoch_counter3, plot_loss3, plot_acc3])
            if finetune:
                st.header("Finetuning whole network")
                col_fine1, col_fine2 = st.columns([1,5])
                with col_fine1:
                    epoch_counter4 = st.empty()
                    epoch_counter4.markdown(f"Epoch 0")
                with col_fine2:
                    progressBar_epoch4 = st.progress(0)
                progressBar_total4 = st.progress(0)
                st.subheader("Loss Monitor")
                plot_loss4 = st.empty()
                st.subheader("Accuracy Monitor")
                plot_acc4 = st.empty()
                train.train_hybridnet(project, num_epochs_hybridnet,
                            None, 'latest', 'all', finetune = True,
                            streamlitWidgets = [progressBar_epoch4,progressBar_total4,
                                                epoch_counter4, plot_loss4, plot_acc4])
            st.balloons()
            time.sleep(1)
            st.experimental_rerun()


def train_center_detect_gui(project, cfg):
    st.header("Train CenterDetect Network")
    st.write("The CenterDetect Network will be used to predict the rough 3D "
                "location of your subject.")
    with st.form("train_center_form"):
        num_epochs = st.number_input("Epochs:",
                    value = cfg.CENTERDETECT.NUM_EPOCHS,
                    min_value = 1, max_value = 1000)
        weights = st.text_input("Weights:", value = "latest",
                    help = "Use 'latest' to load you last saved weights, or "
                                "specify the path to a '.pth' file.")
        submitted = st.form_submit_button("Train")
    if submitted:
        st.header("Training CenterDetect")
        col_center1, col_center2 = st.columns([1,5])
        with col_center1:
            epoch_counter = st.empty()
            epoch_counter.markdown(f"Epoch 0")
        with col_center2:
            progressBar_epoch = st.progress(0)
        progressBar_total = st.progress(0)
        st.subheader("Loss Monitor")
        plot_loss = st.empty()
        st.subheader("Accuracy Monitor")
        plot_acc = st.empty()
        if weights == "":
            weights = None
        elif weights != "latest" and (not os.path.isfile(weights)
                    or weights.split(".") != "pth"):
            st.error("Weights is not a valid file!")
            return
        train.train_efficienttrack('CenterDetect', project,
                    num_epochs, weights,
                    streamlitWidgets = [progressBar_epoch,progressBar_total,
                                        epoch_counter, plot_loss, plot_acc])
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()

def train_keypoint_detect_gui(project, cfg):
    st.header("Train KeypointDetect Network")
    st.write("The KeypointDetect Network will be used to predict the pixel "
                "coordinates of all the joints for a single 2D image.")
    with st.form("train_keypoint_form"):
        num_epochs = st.number_input("Epochs:",
                    value = cfg.KEYPOINTDETECT.NUM_EPOCHS,
                    min_value = 1, max_value = 1000)
        weights = st.text_input("Weights:", value = "latest",
                    help = "Use 'latest' to load you last saved weights, or "
                                "specify the path to a '.pth' file.")
        submitted = st.form_submit_button("Train")
    if submitted:
        st.header("Training KeypointDetect")
        col_center1, col_center2 = st.columns([1,5])
        with col_center1:
            epoch_counter = st.empty()
            epoch_counter.markdown(f"Epoch 0")
        with col_center2:
            progressBar_epoch = st.progress(0)
        progressBar_total = st.progress(0)
        st.subheader("Loss Monitor")
        plot_loss = st.empty()
        st.subheader("Accuracy Monitor")
        plot_acc = st.empty()
        if weights == "":
            weights = None
        elif weights != "latest" and (not os.path.isfile(weights)
                    or weights.split(".") != "pth"):
            st.error("Weights is not a valid file!")
            return
        train.train_efficienttrack('KeypointDetect', project,
                    num_epochs, weights,
                    streamlitWidgets = [progressBar_epoch, progressBar_total,
                                        epoch_counter, plot_loss, plot_acc])
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()

def train_hybridnet_gui(project, cfg):
    st.header("Train Hybridnet")
    st.write("HybridNet is the full network, containing the 2D keypoint "
                "detector as well as the reprojection based 3D multiview network.")
    with st.form("train_ckeypoint_form"):
        num_epochs = st.number_input("Epochs:",
                    value = cfg.HYBRIDNET.NUM_EPOCHS,
                    min_value = 1, max_value = 1000)
        weights_keypoint = st.text_input("Weights KeypointDetect:",
                    value = "latest",
                    help = "Use 'latest' to load you last saved weights, or "
                                "specify the path to a '.pth' file.")
        weights = st.text_input("Weights:", value = "latest",
                    help = "Use 'latest' to load you last saved weights, or "
                                "specify the path to a '.pth' file.")
        mode = st.selectbox('Training Mode', ['3D_only', 'last_layers', 'all'])
        finetune = st.checkbox("Finetune Network",
                    help = "")
        submitted = st.form_submit_button("Train")
    if submitted:
        st.header("Training HybridNet")
        col_center1, col_center2 = st.columns([1,5])
        with col_center1:
            epoch_counter = st.empty()
            epoch_counter.markdown(f"Epoch 0")
        with col_center2:
            progressBar_epoch = st.progress(0)
        progressBar_total = st.progress(0)
        st.subheader("Loss Monitor")
        plot_loss = st.empty()
        st.subheader("Accuracy Monitor")
        plot_acc = st.empty()
        if weights == "":
            weights = None
        elif weights != "latest" and (not os.path.isfile(weights)
                    or weights.split(".") != "pth"):
            st.error("Weights is not a valid file!")
            return
        if weights_keypoint == "":
            weights_keypoint = None
        elif weights_keypoint != "latest" and (not os.path.isfile(weights_keypoint)
                    or weights_keypoint.split(".") != "pth"):
            st.error("Weights KeypointDetect is not a valid file!")
            return
        train.train_hybridnet(project, num_epochs, weights_keypoint, weights,
                    mode, finetune = finetune,
                    streamlitWidgets = [progressBar_epoch,progressBar_total,
                                        epoch_counter, plot_loss, plot_acc])
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()
