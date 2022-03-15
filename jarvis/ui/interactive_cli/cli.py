import os
import time
import cv2
import torch
import inquirer as inq

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
import jarvis.ui.interactive_cli.train_cli as train_cli
import jarvis.ui.interactive_cli.predict_cli as predict_cli
import jarvis.ui.interactive_cli.visualize_cli as visualize_cli



def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def launch_interactive_prompt():
    cls()
    main_menu = [
      inq.List('menu',
            message=f"{CLIColors.OKGREEN}{CLIColors.BOLD}Main Menu{CLIColors.ENDC}",
            choices=['Create Project', 'Train', 'Predict', 'Visualize', 'Exit'],
        )
    ]
    menu = inq.prompt(main_menu)['menu']
    if menu == "Exit":
        return
    elif menu == "Create Project":
        cls()
        create_project()
        launch_interactive_prompt()
    elif menu == 'Train':
        cls()
        train_cli.launch_training_menu()
        launch_interactive_prompt()
    elif menu == "Predict":
        cls()
        predict_cli.launch_prediction_menu()
        launch_interactive_prompt()
    elif menu == "Visualize":
        cls()
        visualize_cli.launch_visualize_menu()
        launch_interactive_prompt()


def create_project():
    questions = [
      inq.Text('name', message="Select a name for the new Project"),
      inq.Text('dataset2D', message="Dataset2D path",
                validate = lambda _, x: (os.path.isdir(x) or x == "")),
      inq.Text('dataset3D', message="Dataset3D path",
                validate = lambda _, x: (os.path.isdir(x) or x == ""))
    ]
    answers = inq.prompt(questions)
    project_name = answers['name']
    dataset2d = answers['dataset2D']
    dataset3d = answers['dataset3D']

    if dataset3d == '':
        print ('[Info] You have not specified a 3D-dataset, you will not be '
                    'able to train the full 3D network!')
    if dataset2d == '':
        dataset2d = dataset3d

    project = ProjectManager()
    project.create_new(
        name = project_name,
        dataset2D_path = dataset2d,
        dataset3D_path = dataset3d)
    print ()
    input("Press Enter to go back to main menu...")
