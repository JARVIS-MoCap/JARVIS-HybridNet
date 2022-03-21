import os
import cv2
import torch
import inquirer as inq

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
import jarvis.predict_interface as predict_interface


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


def get_frame_start_number(video_path):
    predict_full = inq.list_input("Predict for the whole video?", choices=["Yes", "No"])
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
                    and (int(x) > 0 or int(x) == -1) and int(x) < max_num_frames)))
    return frame_start, number_frames


def predict_2D():
    print (f'{CLIColors.OKGREEN}Predict 2D Menu{CLIColors.ENDC}')
    print ('This mode lets you predict the poses on a single video.')
    print ()

    projectManager = ProjectManager()
    projects = projectManager.get_projects()

    project_name = inq.list_input("Select project to load", choices=projects)
    video_path = inq.text("Video Path",
                validate = lambda _, x: (os.path.isfile(x)))
    use_latest_center = inq.list_input("Use most recently saved CenterDetect "
                "weights?", choices=["Yes", "No"])
    if use_latest_center == "Yes":
        weights_center = 'latest'
    else:
        weights_center = inq.zext("Path to CenterDetect '.pth' weights file",
                    validate = lambda _, x: (os.path.isfile(x)
                    and x.split(".")[-1] == 'pth'))
    use_latest_keypoint = inq.list_input("Use most recently saved "
                "KeypointDetect weights?", choices=["Yes", "No"])
    if use_latest_keypoint == "Yes":
        weights_keypoint = 'latest'
    else:
        weights_keypoint = inq.text("Path to KeypointDetect '.pth' weights "
                    "file", validate = lambda _, x: (os.path.isfile(x)
                    and x.split(".")[-1] == 'pth'))
    frame_start, number_frames = get_frame_start_number(video_path)

    skeleton_preset = inq.list_input("Select a Skeleton Preset for Visualization",
                choices=["None", "Hand", "HumanBody", "MonkeyBody", "RodentBody"])
    make_videos = inq.list_input("Make Videos overlayed with the predictions?",
                choices=["Yes", "No"])
    if make_videos == "Yes":
        make_videos = True
    else:
        make_videos = False
    predict_interface.predict2D(project_name, video_path, weights_center,
                weights_keypoint, frame_start, number_frames,
                make_videos, skeleton_preset)

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
    use_latest_center = inq.list_input("Use most recently saved CenterDetect "
                "weights?", choices=["Yes", "No"])
    if use_latest_center == "Yes":
        weights_center = 'latest'
    else:
        weights_center = inq.zext("Path to CenterDetect '.pth' weights file",
                    validate = lambda _, x: (os.path.isfile(x)
                    and x.split(".")[-1] == 'pth'))
    use_latest_hybrid = inq.list_input("Use most recently saved "
                "HybridNet weights?", choices=["Yes", "No"])
    if use_latest_hybrid == "Yes":
        weights_hybridnet = 'latest'
    else:
        weights_hybridnet = inq.text("Path to HybridNet '.pth' weights "
                    "file", validate = lambda _, x: (os.path.isfile(x)
                    and x.split(".")[-1] == 'pth'))

    use_calib_path = inq.list_input("Use calibration that is not used in "
                "trainingset?", choices=["Yes", "No"], default = "No")
    if use_calib_path == "Yes":
        calibration_to_use = inq.text("Enter Calibration Path",
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
            calibration_to_use = None
        else:
            calibration_to_use = inq.list_input("Which calibration should be used?",
                        choices = calibrations)

    example_vid = os.path.join(recordings_path,os.listdir(recordings_path)[0])
    frame_start, number_frames = get_frame_start_number(example_vid)

    skeleton_preset = inq.list_input("Select a Skeleton Preset for Visualization",
                choices=["None", "Hand", "HumanBody", "MonkeyBody", "RodentBody"])
    make_videos = inq.list_input("Make Videos overlayed with the predictions?",
                choices=["Yes", "No"])
    if make_videos == "Yes":
        make_videos = True
    else:
        make_videos = False

    predict_interface.predict3D(project_name, recordings_path, weights_center,
                weights_hybridnet, frame_start, number_frames,
                make_videos, skeleton_preset, dataset_name = calibration_to_use)

    print()
    input("Press Enter to go back to main menu...")
