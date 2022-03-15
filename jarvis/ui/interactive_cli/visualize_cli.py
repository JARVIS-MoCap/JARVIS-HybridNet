import os
import sys
import torch
import cv2
import inquirer as inq
import matplotlib.pyplot as plt

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
import jarvis.visualize_interface as vis_interface
import jarvis.visualize_interface as visualize_interface
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def on_press(event, cancel_3D):
    sys.stdout.flush()
    plt.close()
    if event.key == 'q' or event.key == 'escape':
        cancel_3D['cancel'] = True


def launch_visualize_menu():
    training_menu = [
      inq.List('menu',
            message=f"{CLIColors.OKGREEN}{CLIColors.BOLD}Training Menu{CLIColors.ENDC}",
            choices=['Visualize Dataset2D', 'Visualize Dataset3D', '<< back'])
    ]
    menu = inq.prompt(training_menu)['menu']
    if menu == '<< back':
        return
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
        inq.List('skeleton_preset',
        choices=["None", "Hand", "HumanBody", "MonkeyBody", "RodentBody"],
        message="Select a Skeleton Preset for Visualization"),
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
    skeleton_preset = settings['skeleton_preset']
    if split == "Training":
        set_name = "train"
    else:
        set_name = "val"

    projectManager.load(project_name)
    set = Dataset2D(projectManager.cfg, set=set_name, mode = mode)
    for idx in range(len(set.image_ids)):
        img = vis_interface.visualize_2D_sample(set, mode, idx, skeleton_preset)
        cv2.imshow("", cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)
        if key == 113 or key == 27:
            cv2.destroyAllWindows()
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
        inq.List('skeleton_preset',
        choices=["None", "Hand", "HumanBody", "MonkeyBody", "RodentBody"],
        message="Select a Skeleton Preset for Visualization"),
        inq.List('split',
            message="Load training or validation set?",
            choices=["Training", "Validation"]),
    ]
    settings = inq.prompt(questions)
    project_name = settings['project_name']
    split = settings['split']
    skeleton_preset = settings['skeleton_preset']
    if split == "Training":
        set_name = "train"
    else:
        set_name = "val"

    projectManager.load(project_name)
    set = Dataset3D(projectManager.cfg, set=set_name)
    cancel_3D = {}
    cancel_3D['cancel'] = False
    for idx in range(len(set.image_ids)):
        fig = vis_interface.visualize_3D_sample(set, idx, skeleton_preset)
        fig.canvas.mpl_connect('key_press_event', lambda event: on_press(event, cancel_3D))
        plt.show()
        if cancel_3D['cancel']:
            cancel_3D['cancel'] = False
            cls()
            launch_visualize_menu()
            return
    cls()
    launch_visualize_menu()
