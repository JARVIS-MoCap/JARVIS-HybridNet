import os
import sys
import torch
import cv2
import inquirer as inq
import matplotlib.pyplot as plt

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
import jarvis.analyze_interface as analyze_interface
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D



def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def launch_analyze_menu():
    cls()
    training_menu = [
      inq.List('menu',
            message=f"{CLIColors.OKGREEN}{CLIColors.BOLD}Training Menu{CLIColors.ENDC}",
            choices=['Analyze Validation Data', 'Plot Error Histogram', 'Plot Error per Joint', 'Plot Joint Length Distribution', '<< back'])
    ]
    menu = inq.prompt(training_menu)['menu']
    if menu == '<< back':
        return
    elif menu == "Analyze Validation Data":
        analyze_validation_data()
    elif menu == "Plot Error Histogram":
        plot_error_histogram()
    elif menu == "Plot Error per Joint":
        plot_error_per_joint()
    elif menu == "Plot Joint Length Distribution":
        plot_joint_length_distribution()


def analyze_validation_data():
    cls()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    questions1 = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.List('use_latest_center', choices=["Yes", "No"],
            message="Use most recently saved CenterDetect weights?")
            ]
    settings1 = inq.prompt(questions1)
    project_name = settings1['project_name']
    if settings1['use_latest_center'] == "Yes":
        weights_center = 'latest'
    else:
        weights_center_q = [
            inq.Text('weights_path',
                message="Path to CenterDetect '.pth' weights file",
                validate = lambda _, x: (os.path.isfile(x) and x.split(".")[-1] == 'pth'))
        ]
        weights_center = inq.prompt(weights_center_q)['weights_path']
    use_latest_hybridnet_q = [
        inq.List('use_latest_hybridnet', choices=["Yes", "No"],
            message="Use most recently saved HybridNet weights?")
    ]
    if inq.prompt(use_latest_hybridnet_q)['use_latest_hybridnet'] == "Yes":
        weights_hybridnet = 'latest'
    else:
        weights_hybridnet_q = [
            inq.Text('weights_path',
                message="Path to HybirdNet '.pth' weights file",
                validate = lambda _, x: (os.path.isfile(x) and x.split(".")[-1] == 'pth'))
        ]
        weights_hybridnet = inq.prompt(weights_hybridnet_q)['weights_path']
    analyze_interface.analyze_validation_data(project_name, weights_center,
                weights_hybridnet)

    print ()
    input ("press Enter to continue")
    launch_analyze_menu()


def get_analysis_path():
    project = ProjectManager()
    projects = project.get_projects()
    project_name_q = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects)
    ]
    project_name = inq.prompt(project_name_q)['project_name']
    project.load(project_name)
    cfg = project.get_cfg()
    analysis_path = os.path.join(project.parent_dir,
                project.cfg.PROJECTS_ROOT_PATH, project_name,
                'analysis')
    data_q = [
        inq.List('analysis_path',
            message ="Select Anlysis Set to load",
            choices =os.listdir(analysis_path))
    ]
    analysis_set = inq.prompt(data_q)['analysis_path']
    path = os.path.join(analysis_path, analysis_set)
    return path


def plot_error_histogram():
    cls()
    path = get_analysis_path()
    use_cutoff_q = {
        inq.List('use_cutoff',
            message ="Use Error Cutoff? (Values above cutoff will be grouped in one bin)",
            choices =["Yes", "No"], default = "No")
    }
    cutoff = -1
    if inq.prompt(use_cutoff_q)['use_cutoff'] == "Yes":
        cutoff_q = [
            inq.Text('cutoff',
                message="Cutoff Value",
                validate = lambda _, x: (x.isdigit() and int(x) > 0),
                default = "20")
        ]
        cutoff = int(inq.prompt(cutoff_q)['cutoff'])
    analyze_interface.plot_error_histogram(path, cutoff)
    launch_analyze_menu()


def plot_error_per_joint():
    cls()
    path = get_analysis_path()
    analyze_interface.plot_error_per_joint(path)
    print ()
    input ("press Enter to continue")
    launch_analyze_menu()


def plot_joint_length_distribution():
    cls()
    print ("Not implemented yet, sorry!")
    print ()
    input ("press Enter to continue")
    launch_analyze_menu()
