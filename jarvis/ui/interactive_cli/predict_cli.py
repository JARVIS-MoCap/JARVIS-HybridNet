"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import os
import cv2
import torch
import inquirer as inq

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
from jarvis.prediction.predict3D import predict3D
from jarvis.prediction.predict2D import predict2D
from jarvis.utils.paramClasses import Predict3DParams
from jarvis.utils.paramClasses import Predict2DParams


def cls():
    os.system('cls' if os.name=='nt' else 'clear')



def launch_prediction_menu():
    prediction_menu = [
      inq.List('menu',
            message=f"{CLIColors.OKGREEN}{CLIColors.BOLD}Prediction "
            f"Menu{CLIColors.ENDC}",
            choices=['Predict 3D', 'Predict 2D', '<< back'])
    ]
    menu = inq.prompt(prediction_menu)['menu']
    if menu == '<< back':
        return
    elif menu == "Predict 3D":
        cls()
        predict_3D()
    elif menu == "Predict 2D":
        cls()
        predict_2D()



def predict_2D():
    print (f'{CLIColors.OKGREEN}Predict 2D Menu{CLIColors.ENDC}')
    print ('This mode lets you predict the poses on a single video.')
    print ()

    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    project_name = inq.list_input("Select project to load", choices=projects)
    projectManager.load(project_name)
    cfg = projectManager.get_cfg()

    recording_path = inq.text("Video Path",
                validate = lambda _, x: (os.path.isfile(x)))

    params = Predict2DParams(project_name, recording_path)
    params.trt_mode = get_trt_mode(cfg, "2D")

    if params.trt_mode != 'previous':
        use_latest_center = inq.list_input("Use most recently saved "
                    "CenterDetect weights?", choices=["Yes", "No"])
        if use_latest_center == "Yes":
            params.weights_center_detect = 'latest'
        else:
            params.weights_center_detect = inq.text("Path to CenterDetect "
                        "'.pth' weights file",
                        validate = lambda _, x: (os.path.isfile(x)
                        and x.split(".")[-1] == 'pth'))
        use_latest_keypoint = inq.list_input("Use most recently saved "
                    "KeypointDetect weights?", choices=["Yes", "No"])
        if use_latest_keypoint == "Yes":
            params.weights_keypoint_detect = 'latest'
        else:
            params.weights_keypoint_detect = inq.text("Path to KeypointDetect "
                        "'.pth' weights file",
                        validate = lambda _, x: (os.path.isfile(x)
                        and x.split(".")[-1] == 'pth'))

    params.frame_start, params.number_frames = get_frame_start_number(recording_path)

    predict2D(params)

    print()
    input("Press Enter to go back to main menu...")



def predict_3D():
    print (f'{CLIColors.OKGREEN}Predict 3D Menu{CLIColors.ENDC}')
    print ('This mode lets you predict the poses on a set of recordings.')

    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    project_name = inq.list_input("Select project to load", choices=projects)
    projectManager.load(project_name)
    cfg = projectManager.get_cfg()

    recordings_path = inq.text("Recordings Path",
                validate = lambda _, x: (os.path.isdir(x)))

    params = Predict3DParams(project_name, recordings_path)
    params.trt_mode = get_trt_mode(cfg, "3D")

    if params.trt_mode != 'previous':
        use_latest_center = inq.list_input("Use most recently saved "
                    "CenterDetect weights?", choices=["Yes", "No"])
        if use_latest_center == "Yes":
            params.weights_center_detect = 'latest'
        else:
            params.weights_center_detect = inq.text("Path to CenterDetect "
                        "'.pth' weights file",
                        validate = lambda _, x: (os.path.isfile(x)
                        and x.split(".")[-1] == 'pth'))
        use_latest_hybrid = inq.list_input("Use most recently saved "
                    "HybridNet weights?", choices=["Yes", "No"])
        if use_latest_hybrid == "Yes":
            params.weights_hybridnet = 'latest'
        else:
            params.weights_hybridnet = inq.text("Path to HybridNet "
                        "'.pth' weights file",
                        validate = lambda _, x: (os.path.isfile(x)
                        and x.split(".")[-1] == 'pth'))

    use_calib_path = inq.list_input("Use calibration that is not used in "
                "trainingset?", choices=["Yes", "No"], default = "No")
    if use_calib_path == "Yes":
        params.dataset_name = inq.text("Enter Calibration Path",
                    validate = lambda _, x: (os.path.isdir(x)))
    else:
        dataset_name = cfg.DATASET.DATASET_3D
        if os.path.isabs(dataset_name):
            calib_root_path = os.path.join(dataset_name, 'calib_params')
        else:
            calib_root_path = os.path.join(cfg.PARENT_DIR,
                        cfg.DATASET.DATASET_ROOT_DIR, dataset_name,
                        'calib_params')
        calibrations = os.listdir(calib_root_path)
        if len(calibrations) == 1:
            params.dataset_name = None
        else:
            params.dataset_name = inq.list_input("Which calibration should "
                        "be used?", choices = calibrations)

    example_vid = os.path.join(recordings_path,os.listdir(recordings_path)[0])
    params.frame_start, params.number_frames = \
                get_frame_start_number(example_vid)

    predict3D(params)

    print()
    input("Press Enter to go back to main menu...")



def get_frame_start_number(video_path):
    predict_full = inq.list_input("Predict for the whole video?",
                choices=["Yes", "No"])
    if predict_full == "Yes":
        frame_start = 0
        number_frames = -1
    else:
        cap = cv2.VideoCapture(video_path)
        total_number_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_start = int(inq.text(f"Frame to start predictions at "
                    f"(Max: {int(total_number_frames)})", default = "0",
                    validate = lambda _, x: (x.isdigit() and int(x) >= 0
                    and int(x) < total_number_frames)))
        max_num_frames = total_number_frames - frame_start
        number_frames = int(inq.text(f"Number of frames to predict pose for "
                    f"(Max: {int(max_num_frames)})", default = "-1",
                    validate = lambda _, x: (x.lstrip("-").isdigit()
                    and (int(x) > 0 or int(x) == -1)
                    and int(x) < max_num_frames)))
    return frame_start, number_frames



def check_trt_config(cfg):
    return True



def get_trt_mode(cfg, mode):
    if mode == "2D":
        pt_file_count = 2
        model_dir = "predict2D"
    elif mode == "3D":
        pt_file_count = 3
        model_dir = "predict3D"
    use_trt = inq.list_input("Use TensorRT acceleration?",
                choices=["Yes", "No"], default = "No")
    if use_trt == "Yes":
        search_path = os.path.join(cfg.PARENT_DIR, 'projects',
        cfg.PROJECT_NAME, 'trt-models', model_dir)
        if (os.path.isdir(search_path)
                    and len(os.listdir(search_path)) == pt_file_count):
            use_previous_trt_model = inq.list_input("Use previously saved TRT "
                        "model?", choices=["Yes", "No"], default = "Yes")
            if use_previous_trt_model == "Yes":
                if not check_trt_config(cfg):
                    print ("WARNING: Saved TRT model config different from "
                            "current project config!")
                trt_mode = 'previous'
            else:
                confirm_override = inq.list_input("This will override the old "
                            "TRT model, are you sure?", choices=["Yes", "No"],
                            default = "No")
                if confirm_override == 'Yes':
                    trt_mode = 'new'
                else:
                    if not check_trt_config(cfg):
                        print ("WARNING: Saved TRT model config different "
                                    "from current project config!")
                    trt_mode = 'previous'
        else:
            trt_mode = 'new'
    else:
        trt_mode = 'off'
    return trt_mode
