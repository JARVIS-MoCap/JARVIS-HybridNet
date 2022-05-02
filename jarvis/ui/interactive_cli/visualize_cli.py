import os
import sys
import torch
import cv2
import inquirer as inq
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
from jarvis.visualization.visualize_dataset import visualize_2D_sample, \
            visualize_3D_sample
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.visualization.create_videos3D import create_videos3D
from jarvis.visualization.create_videos2D import create_videos2D
from jarvis.utils.paramClasses import CreateVideos3DParams, CreateVideos2DParams



def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def on_press(event, cancel_3D):
    sys.stdout.flush()
    plt.close()
    if event.key == 'q' or event.key == 'escape':
        cancel_3D['cancel'] = True


def launch_visualize_menu():
    cls()
    training_menu = [
      inq.List('menu',
            message=f"{CLIColors.OKGREEN}{CLIColors.BOLD}Visualize"
                        f" Menu{CLIColors.ENDC}",
            choices=['Create Videos 3D','Create Videos 2D', 'Visualize Dataset2D',
                        'Visualize Dataset3D', '<< back'])
    ]
    menu = inq.prompt(training_menu)['menu']
    if menu == '<< back':
        return
    elif menu == "Create Videos 3D":
        create_videos_3D()
    elif menu == "Create Videos 2D":
        create_videos_2D()
    elif menu == "Visualize Dataset2D":
        visualize_2D()
    elif menu == "Visualize Dataset3D":
        visualize_3D()


def visualize_2D():
    cls()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    questions = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.List('split',
            message="Load training or validation set?",
            choices=["Training", "Validation"]),
        inq.List('mode',
            message="Select Mode",
            choices=["CenterDetect", "KeypointDetect"])
    ]
    settings = inq.prompt(questions)
    project_name = settings['project_name']
    split = settings['split']
    mode = settings['mode']
    if split == "Training":
        set_name = "train"
    else:
        set_name = "val"

    projectManager.load(project_name)
    set = Dataset2D(projectManager.cfg, set=set_name, mode = mode)
    cancel_2D = {}
    cancel_2D['cancel'] = False
    for idx in range(len(set.image_ids)):
        fig = visualize_2D_sample(set, mode, idx)
        fig.canvas.mpl_connect('key_press_event',
                    lambda event: on_press(event, cancel_2D))
        plt.show()
        if cancel_2D['cancel']:
            cancel_2D['cancel'] = False
            cls()
            launch_visualize_menu()
            return
    cv2.destroyAllWindows()
    cls()
    launch_visualize_menu()


def visualize_3D():
    cls()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    questions = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.List('split',
            message="Load training or validation set?",
            choices=["Training", "Validation"]),
    ]
    settings = inq.prompt(questions)
    project_name = settings['project_name']
    split = settings['split']
    if split == "Training":
        set_name = "train"
    else:
        set_name = "val"

    projectManager.load(project_name)
    set = Dataset3D(projectManager.cfg, set=set_name)
    cancel_3D = {}
    cancel_3D['cancel'] = False
    for idx in range(len(set.image_ids)):
        fig = visualize_3D_sample(set, idx)
        fig.canvas.mpl_connect('key_press_event',
                    lambda event: on_press(event, cancel_3D))
        plt.show()
        if cancel_3D['cancel']:
            cancel_3D['cancel'] = False
            cls()
            launch_visualize_menu()
            return
    cls()
    launch_visualize_menu()


def create_videos_3D():
    print (f'{CLIColors.OKGREEN}Create Videos 3D Menu{CLIColors.ENDC}')
    print ('This mode lets you create annotated videos from one of your '
                '3D predictions.')

    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    project_name = inq.list_input("Select project to load", choices=projects)
    projectManager.load(project_name)
    cfg = projectManager.get_cfg()

    data_csv = None
    prediction_path = get_prediction_path(cfg, '3D')
    if prediction_path != None:
        data_csv = get_data_csv(prediction_path)

    if data_csv == None:
        print ("No '.csv' file containing predictions found! Aborting...")
        print()
        input("Press Enter to go back to main menu...")
        return

    with open(os.path.join(prediction_path, 'info.yaml')) as file:
        yaml = YAML()
        info_yaml = yaml.load(file)
        recordings_path = info_yaml['recording_path']
        dataset_name = info_yaml['dataset_name']
        frame_start = info_yaml['frame_start']
        number_frames = info_yaml['number_frames']

    params = CreateVideos3DParams(project_name, recordings_path, data_csv)

    params.dataset_name = dataset_name
    params.frame_start = frame_start
    params.number_frames = number_frames

    cameras = []
    videos = os.listdir(params.recording_path)
    for video in os.listdir(params.recording_path):
        cameras.append(video.split('.')[0])
    params.video_cam_list = inq.checkbox("Select cameras to create videos "
                "with", choices=cameras, default = cameras)

    create_videos3D(params)

    print()
    input("Press Enter to go back to main menu...")


def create_videos_2D():
    print (f'{CLIColors.OKGREEN}Create Videos 2D Menu{CLIColors.ENDC}')
    print ('This mode lets you create an annotated from one of your '
                '2D predictions.')

    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    project_name = inq.list_input("Select project to load", choices=projects)
    projectManager.load(project_name)
    cfg = projectManager.get_cfg()

    data_csv = None
    prediction_path = get_prediction_path(cfg, '2D')
    if prediction_path != None:
        data_csv = get_data_csv(prediction_path)

    if data_csv == None:
        print ("No '.csv' file containing predictions found! Aborting...")
        print()
        input("Press Enter to go back to main menu...")
        return

    with open(os.path.join(prediction_path, 'info.yaml')) as file:
        yaml = YAML()
        info_yaml = yaml.load(file)
        recording_path = info_yaml['recording_path']
        frame_start = info_yaml['frame_start']
        number_frames = info_yaml['number_frames']

    params = CreateVideos2DParams(project_name, recording_path, data_csv)

    params.frame_start = frame_start
    params.number_frames = number_frames

    create_videos2D(params)

    print()
    input("Press Enter to go back to main menu...")


def get_prediction_path(cfg, mode):
    predict_path = os.path.join(cfg.PARENT_DIR,
                cfg.PROJECTS_ROOT_PATH, cfg.PROJECT_NAME,
                'predictions', f'predictions{mode}')

    if (not os.path.exists(predict_path)) or len(os.listdir(predict_path)) == 0:
        print (f"No predictions created yet. Please run Predict{mode} first!")
        return None

    prediction = inq.list_input("Select Prediction to load",
                choices = sorted(os.listdir(predict_path))[::-1])
    path = os.path.join(predict_path, prediction)
    return path


def get_data_csv(path):
    files = os.listdir(path)
    files = [file for file in files if file.split(".")[-1] == 'csv']
    if len(files) == 1:
        data_csv = os.path.join(path, files[0])
        return data_csv
    elif len(files) > 1:
        data_csv = inq.list_input("Select prediction savefile to use",
                    choices=files)
        return os.path.join(path, data_csv)


    return None
