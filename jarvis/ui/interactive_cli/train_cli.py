import os
import torch
import inquirer as inq

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
import jarvis.train_interface as train_interface
import jarvis.utils.clp as clp
from jarvis.utils.utils import get_available_pretrains

def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def check_gpus():
    if (torch.cuda.device_count() == 0):
        clp.error('Aborting! You can only run this on a computer '
                    'that has at least one GPU...')
        print ()
        input("Press Enter to go back to main menu...")
        return False
    return True


def launch_training_menu():
    menu_items = ['Train All', 'Train CenterDetect', 'Train KeypointDetect',
            'Train HybridNet', '<< back']
    menu = inq.list_input(f"{CLIColors.OKGREEN}{CLIColors.BOLD}Training "
                f"Menu{CLIColors.ENDC}", choices = menu_items)
    if menu == '<< back':
        return
    elif menu == "Train All":
        train_all()
    elif menu == "Train CenterDetect":
        train_center_detect()
    elif menu == "Train KeypointDetect":
        train_keypoint_detect()
    elif menu == "Train HybridNet":
        train_hybridnet()


def get_project_and_pretrain():
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    project_name = inq.list_input("Select project to load", choices=projects)
    available_pretrains = get_available_pretrains(projectManager.parent_dir)
    weights = inq.list_input("Select pretrain to be used",
                choices=['None'] + available_pretrains)

    if weights == 'None':
        specify_path = inq.list_input("Specify weights to load before training?",
                    choices=["Yes", "No"], default="No")
        if specify_path == "Yes":
            weights = inq.text("Path to '.pth' weights file",
                        validate = lambda _, x: (os.path.isfile(x)
                        and x.split(".")[-1] == 'pth'))
    return project_name, weights


def train_center_detect():
    cls()
    print (f'{CLIColors.OKGREEN}{CLIColors.BOLD}Training CenterDetect!'
                f'{CLIColors.ENDC}')
    print ('You can change all network training setttings in the projects '
                '\'config.yaml\'.')
    print()
    project_name, weights = get_project_and_pretrain()
    projectManager = ProjectManager()
    if not projectManager.load(project_name):
        clp.error(f"Could not load Project {project_name}!")
        print ()
        input("Press Enter to go back to main menu...")
        return
    cfg = projectManager.cfg
    num_epochs = int(inq.text("Set Number of Epochs to train for",
                default = cfg.CENTERDETECT.NUM_EPOCHS,
                validate = lambda _, x: (x.isdigit() and int(x) > 0)))
    if not check_gpus():
        return
    train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs, weights)
    print ()
    clp.success('Training finished! Your CenterDetect network is '
                'ready for prediction, have fun :)')
    print ()
    input("Press Enter to go back to main menu...")



def train_keypoint_detect():
    cls()
    print (f'{CLIColors.OKGREEN}{CLIColors.BOLD}Training KeypointDetect!'
                f'{CLIColors.ENDC}')
    print ('You can change all network training setttings in the projects '
                '\'config.yaml\'.')
    print()
    project_name, weights = get_project_and_pretrain()
    projectManager = ProjectManager()
    if not projectManager.load(project_name):
        clp.error(f"Could not load Project {project_name}!")
        print ()
        input("Press Enter to go back to main menu...")
        return
    cfg = projectManager.cfg

    num_epochs = int(inq.text("Set Number of Epochs to train for",
                default = cfg.KEYPOINTDETECT.NUM_EPOCHS,
                validate = lambda _, x: (x.isdigit() and int(x) > 0)))
    if not check_gpus():
        return

    train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs, weights)
    print ()
    clp.success('{Training finished! Your KeypointDetect network is '
                'ready for prediction, have fun :)')
    print ()
    input("Press Enter to go back to main menu...")


def train_hybridnet():
    cls()
    print (f'{CLIColors.OKGREEN}{CLIColors.BOLD}Training HybridNet!'
                f'{CLIColors.ENDC}')
    print ('You can change all network training setttings in the projects '
                '\'config.yaml\'.')
    print()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    project_name = inq.list_input("Select project to load", choices=projects)
    if not projectManager.load(project_name):
        clp.error(f"Could not load Project {project_name}!")
        print ()
        input("Press Enter to go back to main menu...")
        return

    use_latest_keypoint = inq.list_input("Use most recently saved "
                "KeypointDetect weights?", choices=["Yes", "No"])
    if use_latest_keypoint == "Yes":
        weights_keypoint_detect = 'latest'
    else:
        weights_keypoint_detect = inq.text("Path to KeypointDetect "
                    "'.pth' weights file",
                    validate = lambda _, x: ((os.path.isfile(x)
                    and x.split(".")[-1] == 'pth') or x == ""))

    if weights_keypoint_detect == "":
        weights_keypoint_detect = None
        weights_hybridnet = inq.text("Path to '.pth' Hybridnet weights file",
                    validate = lambda _, x: ((os.path.isfile(x)
                    and x.split(".")[-1] == 'pth') or x == ""))
        if weights_hybridnet == "":
            weights_hybridnet = None
    else:
        weights_hybridnet = None

    num_epochs = int(inq.text("Set Number of Epochs to train for",
                default = projectManager.cfg.HYBRIDNET.NUM_EPOCHS,
                validate = lambda _, x: (x.isdigit() and int(x) > 0)))
    mode = inq.list_input("Select training mode (only use mode different from "
                "3D_only if you know what you're doing)", choices= ['3D_only',
                'last_layers', 'bifpn', 'all'])
    if mode == '3D_only':
        finetune = False
    else:
        finetune = True
    if not check_gpus():
        return

    train_interface.train_hybridnet(project_name, num_epochs,
                weights_keypoint_detect, weights_hybridnet, mode, finetune)
    print ()
    clp.success('Training finished! Your HybridNet is '
                'ready for prediction, have fun :)')
    print ()
    input("Press Enter to go back to main menu...")


def train_all():
    cls()
    print (f'{CLIColors.OKGREEN}{CLIColors.BOLD}Training Full Network Stack!'
                f'{CLIColors.ENDC}')
    clp.info('You can change all network training settings in the projects '
                '\'config.yaml\'.')
    print()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    available_pretrains = get_available_pretrains(projectManager.parent_dir)
    if not projectManager.load(project_name):
        clp.error(f"Could not load Project {project_name}!")
        print ()
        input("Press Enter to go back to main menu...")
        return
    cfg = projectManager.cfg

    questions = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.List('pretrain',
            message="Select pretrain to be used",
            choices=['None'] + available_pretrains),
        inq.Text('num_epochs_center',
                    message="Set Number of Epochs for CenterDetect",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = cfg.CENTERDETECT.NUM_EPOCHS),
        inq.Text('num_epochs_keypoint',
                    message="Set Number of Epochs for KeypointDetect",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = cfg.KEYPOINTDETECT.NUM_EPOCHS),
        inq.Text('num_epochs_hybridnet',
                    message="Set Number of Epochs for HybridNet",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = cfg.HYBRIDNET.NUM_EPOCHS),
    ]
    settings = inq.prompt(questions)
    if not check_gpus():
        return

    project_name = settings['project_name']
    num_epochs_center = int(settings['num_epochs_center'])
    num_epochs_keypoint = int(settings['num_epochs_keypoint'])
    num_epochs_hybridnet = int(settings['num_epochs_hybridnet'])
    pretrain = settings['pretrain']

    print ()
    print (f'Training all newtorks for project {project_name} for '
                f'{num_epochs_center} epochs!')
    print (f'First training CenterDetect for {num_epochs_center} epochs...')
    trained = train_interface.train_efficienttrack('CenterDetect',
                project_name, num_epochs_center, pretrain)
    if not trained:
            print ()
            input("Press Enter to go back to main menu...")
            return
    print (f'Training KeypointDetect for {num_epochs_keypoint} epochs...')
    trained = train_interface.train_efficienttrack('KeypointDetect',
                project_name, num_epochs_keypoint, pretrain)
    if not trained:
        print ()
        input("Press Enter to go back to main menu...")
        return
    print (f'Training 3D section of HybridNet for {num_epochs_hybridnet} '
                f'epochs...')
    train_interface.train_hybridnet(project_name, num_epochs_hybridnet,
                'latest', None, '3D_only')
    print ()
    clp.success('{Training finished! Your networks are '
                f'ready for prediction, have fun :)')
    print ()
    input("Press Enter to go back to main menu...")
