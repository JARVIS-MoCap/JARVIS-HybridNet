import os
import torch
from tensorboard import program
import inquirer as inq

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
from jarvis.dataset.dataset2D import Dataset2D
from jarvis.dataset.dataset3D import Dataset3D
from jarvis.efficienttrack.efficienttrack import EfficientTrack
from jarvis.hybridnet.hybridnet import HybridNet
from jarvis.prediction.predict2D import predictPosesVideo
from jarvis.prediction.predict3D import predictPosesVideos, load_reprojection_tools
from jarvis.visualization.time_slices import plot_slices
import jarvis.train_interface as train_interface
import jarvis.predict_interface as predict_interface
import jarvis.visualize_interface as visualize

def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def launch_interactive_prompt():
    cls()
    main_menu = [
      inq.List('menu',
            message="Main Menu",
            choices=['Create Project', 'Train', 'Predict', 'Visualize', 'Exit'],
        )
    ]
    menu = inq.prompt(main_menu)['menu']
    if menu == "Exit":
        return
    elif menu == "Create Project":
        cls()
        create_project()
    elif menu == 'Train':
        cls()
        launch_training_menu()
    else:
        launch_interactive_prompt()

def create_project():
    questions = [
      inq.Text('name', message="Select a name for the new Project:"),
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
    launch_interactive_prompt()


def launch_training_menu():
    training_menu = [
      inq.List('menu',
            message="Training Menu",
            choices=['Train All', 'Train CenterDetect', 'Train KeypointDetect',
            'Train HybridNet', '<< back'])
    ]
    menu = inq.prompt(training_menu)['menu']
    if menu == '<< back':
        launch_interactive_prompt()
        return
    elif menu == "Train All":
        train_all()
    elif menu == "Train CenterDetect":
        train_center_detect()
    elif menu == "Train KeypointDetect":
        train_keypoint_detect()
    elif menu == "Train HybridNet":
        train_hybridnet()


def train_center_detect():
    cls()
    print (f'{CLIColors.OKGREEN}Training CenterDetect!{CLIColors.ENDC}')
    print()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    questions1 = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.List('pretrain',
            message="Select pretrain to be used",
            choices=['None', 'HumanHand', 'MonkeyHand', 'HumanBody',
            'RatBody', 'MouseBody']),
    ]
    settings1 = inq.prompt(questions1)
    project_name = settings1['project_name']
    weights = settings1['pretrain']
    if weights == 'None':
        question_weights = [
            inq.List('specify_path', message="Specify '.pth' weights to load before training?",
            choices=["Yes", "No"], default="No")
        ]
        if (inq.prompt(question_weights)['specify_path'] == "Yes"):
            question_weights_path = [
                inq.Text('weights_path',
                            message="Path to '.pth' weights file",
                            validate = lambda _, x: (os.path.isfile(x) and x.split(".")[-1] == 'pth'))
            ]
            weights = inq.prompt(question_weights_path)['weights_path']
    print (weights)

# @click.command(name='centerDetect')
# @click.option('--num_epochs', default=None,
#             type =click.IntRange(min=1),
#             help='Number of Epochs, try 100 or more for very small datasets.')
# @click.option('--weights_path', default=None,
#             help='Weights to load before training. You have to specify the '
#             'path to a specific \'.pth\' file')
# @click.option('--pretrained_weights', default='None',
#         type=click.Choice(['None', 'EcoSet', 'HumanHand', 'MonkeyHand',
#         'HumanBody', 'RatBody', 'MouseBody'], case_sensitive=False),
#         prompt='Which pretrain do you want to use (Select \'None\' if you '
#         'specified a \'weights_path\')?\n',
#         help='Pretrain to load before training. Select either \'EcoSet\' or '
#         'one of the pretrained pose estimators available')
# @click.option('--gpu', default=None,
#             type=click.IntRange(min=0, max =torch.cuda.device_count()-1),
#             help='Number of the GPU to be used')
# @click.argument('project_name')
# def train_center_detect(project_name, num_epochs, weights_path,
#             pretrained_weights, gpu):
#     set_gpu_environ(gpu)
#     if weights_path != None:
#         weights = weights_path
#     elif pretrained_weights != 'None':
#         weights = pretrained_weights
#     else:
#         weights = None
#     train_interface.train_efficienttrack('CenterDetect', project_name,
#                 num_epochs, weights)

def train_all():
    cls()
    print (f'{CLIColors.OKGREEN}Training Full Network Stack!{CLIColors.ENDC}')
    print()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()

    questions = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.List('pretrain',
            message="Select pretrain to be used",
            choices=['None', 'HumanHand', 'MonkeyHand', 'HumanBody',
            'RatBody', 'MouseBody']),
        inq.List('finetune', message="Finetune HybridNet? "
                    "(Slow and GPU RAM-hungry)", choices=["Yes", "No"], default="No"),
        inq.Text('num_epochs_center',
                    message="Set Number of Epochs for CenterDetect",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = 50),
        inq.Text('num_epochs_keypoint',
                    message="Set Number of Epochs for KeypointDetect",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = 100),
        inq.Text('num_epochs_hybridnet',
                    message="Set Number of Epochs for HybridNet",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = 50),
        inq.List('gpu',
            message="Index of GPU to be used",
            choices=range(torch.cuda.device_count()))
    ]
    settings = inq.prompt(questions)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(settings['gpu'])
    project_name = settings['project_name']
    num_epochs_center = int(settings['num_epochs_center'])
    num_epochs_keypoint = int(settings['num_epochs_keypoint'])
    num_epochs_hybridnet = int(settings['num_epochs_hybridnet'])
    pretrain = settings['pretrain']

    print ()
    print (f'Training all newtorks for project {project_name} for '
                f'{num_epochs_center} epochs!')
    print (f'First training CenterDetect for {num_epochs_center} epochs...')
    trained = train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs_center, pretrain)
    if not trained:
            print ()
            input("Press Enter to go back to main menu...")
            launch_interactive_prompt()
            return
    print (f'Training KeypointDetect for {num_epochs_keypoint} epochs...')
    trained = train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs_keypoint, pretrain)
    if not trained:
        print ()
        input("Press Enter to go back to main menu...")
        launch_interactive_prompt()
        return
    print (f'Training 3D section of HybridNet for {num_epochs_hybridnet} '
                f'epochs...')
    train_interface.train_hybridnet(project_name, num_epochs_hybridnet,
                'latest', None, '3D_only')
    if settings['finetune']:
        print (f'Finetuning complete HybridNet for {num_epochs_hybridnet} '
                    f'epochs...')
        train_interface.train_hybridnet(project_name, num_epochs_hybridnet,
                    None, 'latest', 'all', finetune = True)
    print ()
    print (f'{CLIColors.OKGREEN}Training finished! You networks are '
                f'ready for prediction, have fun :){CLIColors.ENDC}')
    print ()
    input("Press Enter to go back to main menu...")
    launch_interactive_prompt()
