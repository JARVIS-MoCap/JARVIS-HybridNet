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
            message=f"{CLIColors.OKGREEN}{CLIColors.BOLD}Prediction Menu{CLIColors.ENDC}",
            choices=['Predict 2D', 'Predict 3D', '<< back'])
    ]
    menu = inq.prompt(prediction_menu)['menu']
    if menu == '<< back':
        return
    elif menu == "Predict 2D":
        cls()
        predict_2D()
    elif menu == "Predict 3D":
        cls()
        predict_3D()


def predict_2D():
    print (f'{CLIColors.OKGREEN}Predict 2D Menu{CLIColors.ENDC}')
    print ('This mode lets you predict the poses on a single video.')
    print ()

    projectManager = ProjectManager()
    projects = projectManager.get_projects()

    questions1 = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.Text('video_path', message="Video Path",
            validate = lambda _, x: (os.path.isfile(x))),
        inq.List('use_latest_center', choices=["Yes", "No"],
            message="Use most recently saved CenterDetect weights?"),
    ]
    settings1 = inq.prompt(questions1)
    project_name = settings1['project_name']
    video_path = settings1['video_path']
    if settings1['use_latest_center'] == "Yes":
        weights_center = 'latest'
    else:
        weights_center_q = [
            inq.Text('weights_path',
                message="Path to CenterDetect '.pth' weights file",
                validate = lambda _, x: (os.path.isfile(x) and x.split(".")[-1] == 'pth'))
        ]
        weights_center = inq.prompt(weights_center_q)['weights_path']
    use_latest_keypoint_q = [
        inq.List('use_latest_keypoint', choices=["Yes", "No"],
            message="Use most recently saved KeypointDetect weights?")
    ]
    if inq.prompt(use_latest_keypoint_q)['use_latest_keypoint'] == "Yes":
        weights_keypoint = 'latest'
    else:
        weights_keypoints_q = [
            inq.Text('weights_path',
                message="Path to KeypointDetect '.pth' weights file",
                validate = lambda _, x: (os.path.isfile(x) and x.split(".")[-1] == 'pth'))
        ]
        weights_keypoint = inq.prompt(weights_keypoints_q)['weights_path']

    cap = cv2.VideoCapture(video_path)
    total_number_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_start_q = [
        inq.Text('frame_start',
            message=f"Frame to start predictions at (Max: {int(total_number_frames)})",
            validate = lambda _, x: (x.isdigit() and int(x) >= 0 and int(x) < total_number_frames),
            default = "0")
    ]
    frame_start = int(inq.prompt(frame_start_q)['frame_start'])
    max_num_frames = total_number_frames - frame_start
    number_frames_q = [
        inq.Text('number_frames',
            message=f"Number of frames to predict pose for (Max: {int(max_num_frames)})",
            validate = lambda _, x: (x.lstrip("-").isdigit() and (int(x) > 0 or int(x) == -1) and int(x) < max_num_frames),
            default = "-1")
    ]
    number_frames = int(inq.prompt(number_frames_q)['number_frames'])
    questions2 = [
        inq.List('skeleton_preset', choices=["None", "Hand", "HumanBody", "MonkeyBody", "RodentBody"],
            message="Select a Skeleton Preset for Visualization"),
        inq.List('make_videos', choices=["Yes", "No"],
            message="Make Videos overlayed with the predictions?")
    ]
    settings2 = inq.prompt(questions2)
    skeleton_preset = settings2['skeleton_preset']
    if settings2['make_videos'] == "Yes":
        make_videos = True
    else:
        make_videos = False
    load = projectManager.load(project_name)
    predict_interface.predict2D(project_name, video_path, weights_center,
                weights_keypoint, frame_start, number_frames,
                make_videos, skeleton_preset)

    print()
    input("Press Enter to go back to main menu...")


def predict_3D():
    print (f'{CLIColors.OKGREEN}Predict 3D Menu{CLIColors.ENDC}')
    print ('This mode lets you predict the poses on a set of recordings.')
    print()
    input("Press Enter to go back to main menu...")
