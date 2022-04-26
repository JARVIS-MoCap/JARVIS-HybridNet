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
        pretrain = st.selectbox('Pretraining to use',
                    ['None', 'EcoSet', 'MonkeyHand', 'HumanHand', 'HumanBody','RatBody', 'MouseBody'])
        finetune = st.checkbox("Finetune Network", value = False)
        submitted = st.form_submit_button("Train")
    if submitted:
        if not check_config_all(project, cfg):
            return
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
            trained = train.train_efficienttrack('CenterDetect', project,
                        num_epochs_center, pretrain,
                        streamlitWidgets = [progressBar_epoch1, progressBar_total1,
                                            epoch_counter1, plot_loss1, plot_acc1])
            if not trained:
                st.error("Could not find pretraining weights, aborting training!")#
                return

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
            trained = train.train_efficienttrack('KeypointDetect', project,
                        num_epochs_keypoint, pretrain,
                        streamlitWidgets = [progressBar_epoch2,progressBar_total2,
                                            epoch_counter2, plot_loss2, plot_acc2])
            if not trained:
                st.error("Could not find pretraining weights, aborting training!")#
                return

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
        if not check_config_center_detect(project, cfg):
            return
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
                    or weights.split(".")[-1] != "pth"):
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
        if not check_config_keypoint_detect(project, cfg):
            return
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
                    or weights.split(".")[-1] != "pth"):
            st.error("Weights is not a valid file!")
            return
        train.train_efficienttrack('KeypointDetect', project,
                    num_epochs, weights,
                    streamlitWidgets = [progressBar_epoch, progressBar_total,
                                        epoch_counter, plot_loss, plot_acc])
        st.balloons()
        #time.sleep(1)
        #st.experimental_rerun()

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
        if not check_config_hybridnet(project, cfg):
            return
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
                    or weights.split(".")[-1] != "pth"):
            st.error("Weights is not a valid file!")
            return
        if weights_keypoint == "":
            weights_keypoint = None
        elif weights_keypoint != "latest" and (not os.path.isfile(weights_keypoint)
                    or weights_keypoint.split(".")[-1] != "pth"):
            st.error("Weights KeypointDetect is not a valid file!")
            return
        train.train_hybridnet(project, num_epochs, weights_keypoint, weights,
                    mode, finetune = finetune,
                    streamlitWidgets = [progressBar_epoch,progressBar_total,
                                        epoch_counter, plot_loss, plot_acc])
        st.balloons()
        time.sleep(1)
        st.experimental_rerun()


def check_config_all(project, cfg):
    if not check_dataset2D(project, cfg):
        return False
    if not check_dataset3D(project, cfg):
        return False
    if not check_center_detect(project, cfg):
        return False
    if not check_keypoint_detect(project, cfg):
        return False
    if not check_hybridnet(project, cfg):
        return False
    return True


def check_config_center_detect(project, cfg):
    if not check_dataset2D(project, cfg):
        return False
    if not check_center_detect(project, cfg):
        return False
    return True


def check_config_keypoint_detect(project, cfg):
    if not check_dataset2D(project, cfg):
        return False
    if not check_keypoint_detect(project, cfg):
        return False
    return True


def check_config_hybridnet(project, cfg):
    if not check_dataset3D(project, cfg):
        return False
    if not check_hybridnet(project, cfg):
        return False
    return True


def check_dataset2D(project, cfg):
    if os.path.isabs(cfg.DATASET.DATASET_2D):
        dataset2D_path = cfg.DATASET.DATASET_2D
    else:
        dataset2D_path = os.path.join(cfg.PARENT_DIR, cfg.DATASET.DATASET_ROOT_DIR, cfg.DATASET.DATASET_2D)
    if not os.path.isdir(dataset2D_path):
        st.error("Dataset2D does not exist, please check path!")
        return False
    return True

def check_dataset3D(project, cfg):
    if os.path.isabs(cfg.DATASET.DATASET_3D):
        dataset3D_path = cfg.DATASET.DATASET_3D
    else:
        dataset3D_path = os.path.join(cfg.PARENT_DIR, cfg.DATASET.DATASET_ROOT_DIR, cfg.DATASET.DATASET_3D)
    if not os.path.isdir(dataset3D_path):
        st.error("Dataset3D does not exist, please check path!")
        return False
    return True

def check_center_detect(project, cfg):
    if cfg.CENTERDETECT.COMPOUND_COEF < 0 or cfg.CENTERDETECT.COMPOUND_COEF > 8:
        st.error("CenterDetect Compound Coefficient has to be in valid range of 0-8!")
        return False
    if cfg.CENTERDETECT.BATCH_SIZE <= 0:
        st.error("CenterDetect Batch Size has to be bigger than 0!")
        return False
    if cfg.CENTERDETECT.MAX_LEARNING_RATE <= 0:
        st.error("CenterDetect Learning Rate has to be bigger than 0!")
        return False
    if cfg.CENTERDETECT.NUM_EPOCHS <= 0:
        st.error("CenterDetect Number of Epochs has to be bigger than 0!")
        return False
    if cfg.CENTERDETECT.CHECKPOINT_SAVE_INTERVAL <= 0:
        st.error("CenterDetect Checkpoint Save Interval has to be bigger than 0!")
        return False
    if cfg.CENTERDETECT.IMAGE_SIZE <= 0 or cfg.CENTERDETECT.IMAGE_SIZE % 64 != 0:
        st.error("CenterDetect Image Size has to be bigger than 0 and divisible by 64!")
        return False
    return True


def check_keypoint_detect(project, cfg):
    if cfg.KEYPOINTDETECT.COMPOUND_COEF < 0 or cfg.KEYPOINTDETECT.COMPOUND_COEF > 8:
        st.error("KeypointDetect Compound Coefficient has to be in valid range of 0-8!")
        return False
    if cfg.KEYPOINTDETECT.BATCH_SIZE <= 0:
        st.error("KeypointDetect Batch Size has to be bigger than 0!")
        return False
    if cfg.KEYPOINTDETECT.MAX_LEARNING_RATE <= 0:
        st.error("KeypointDetect Learning Rate has to be bigger than 0!")
        return False
    if cfg.KEYPOINTDETECT.NUM_EPOCHS <= 0:
        st.error("KeypointDetect Number of Epochs has to be bigger than 0!")
        return False
    if cfg.KEYPOINTDETECT.CHECKPOINT_SAVE_INTERVAL <= 0:
        st.error("KeypointDetect Checkpoint Save Interval has to be bigger than 0!")
        return False
    if cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE <= 0 or cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE % 64 != 0:
        st.error("KeypointDetect Bounding Box Size has to be bigger than 0 and divisible by 64!")
        return False
    if cfg.KEYPOINTDETECT.NUM_JOINTS <= 0:
        st.error("KeypointDetect Number of JOints has to be bigger than 0!")
        return False
    return True


def check_hybridnet(project, cfg):
    if cfg.HYBRIDNET.BATCH_SIZE <= 0:
        st.error("HybridNet Batch Size has to be bigger than 0!")
        return False
    if cfg.HYBRIDNET.MAX_LEARNING_RATE <= 0:
        st.error("HybridNet Learning Rate has to be bigger than 0!")
        return False
    if cfg.HYBRIDNET.NUM_EPOCHS <= 0:
        st.error("HybridNet Number of Epochs has to be bigger than 0!")
        return False
    if cfg.HYBRIDNET.CHECKPOINT_SAVE_INTERVAL <= 0:
        st.error("HybridNet Checkpoint Save Interval has to be bigger than 0!")
        return False
    if cfg.HYBRIDNET.NUM_CAMERAS < 2:
        st.error("HybridNet Number of Cameras has to be at least 2!")
        return False
    if cfg.HYBRIDNET.ROI_CUBE_SIZE <= 0:
        st.error("HybridNet ROI Cube Size has to be bigger than 0!")
        return False
    if cfg.HYBRIDNET.GRID_SPACING <= 0:
        st.error("HybridNet Grid Spacing has to be bigger than 0!")
        return False
    if cfg.HYBRIDNET.ROI_CUBE_SIZE % (cfg.HYBRIDNET.GRID_SPACING*8) != 0:
        st.error("HybirdNet ROI_CUBE_SIZE has to be divisible by 4 * GRID_SPACING!")
        return False
    return True
