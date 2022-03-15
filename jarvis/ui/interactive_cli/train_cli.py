import os
import torch
import inquirer as inq

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors
import jarvis.train_interface as train_interface


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def check_gpus():
    if (torch.cuda.device_count() > 1):
        gpu_q = [
            inq.List('gpu',
                message="Index of GPU to be used",
                choices=range(torch.cuda.device_count()))
        ]
        gpu_id = inq.prompt(gpu_q)['gpu']
        os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu_id)
    if (torch.cuda.device_count() == 0):
        print (f'{CLIColors.FAIL}Aborting! You can only run this on a computer that has at least one GPU...{CLIColors.ENDC}')
        print ()
        input("Press Enter to go back to main menu...")
        return False
    return True


def launch_training_menu():
    training_menu = [
      inq.List('menu',
            message=f"{CLIColors.OKGREEN}{CLIColors.BOLD}Training Menu{CLIColors.ENDC}",
            choices=['Train All', 'Train CenterDetect', 'Train KeypointDetect',
            'Train HybridNet', '<< back'])
    ]
    menu = inq.prompt(training_menu)['menu']
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


def train_center_detect():
    cls()
    print (f'{CLIColors.OKGREEN}Training CenterDetect!{CLIColors.ENDC}')
    print ('You can change all network training setttings in the projects \'config.yaml\'.')
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
            inq.List('specify_path', message="Specify weights to load before training?",
            choices=["Yes", "No"], default="No")
        ]
        if (inq.prompt(question_weights)['specify_path'] == "Yes"):
            question_weights_path = [
                inq.Text('weights_path',
                            message="Path to '.pth' weights file",
                            validate = lambda _, x: (os.path.isfile(x) and x.split(".")[-1] == 'pth'))
            ]
            weights = inq.prompt(question_weights_path)['weights_path']
    epoch_q = [
        inq.Text('num_epochs',
                    message="Set Number of Epochs to train for",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = 50),
    ]
    num_epochs = inq.prompt(epoch_q)['num_epochs']
    if not check_gpus():
        return
    train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs, weights)
    print ()
    print (f'{CLIColors.OKGREEN}Training finished! You CenterDetect network is '
                f'ready for prediction, have fun :){CLIColors.ENDC}')
    print ()
    input("Press Enter to go back to main menu...")



def train_keypoint_detect():
    cls()
    print (f'{CLIColors.OKGREEN}Training KeypointDetect!{CLIColors.ENDC}')
    print ('You can change all network training setttings in the projects \'config.yaml\'.')
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
            inq.List('specify_path', message="Specify weights to load before training?",
            choices=["Yes", "No"], default="No")
        ]
        if (inq.prompt(question_weights)['specify_path'] == "Yes"):
            question_weights_path = [
                inq.Text('weights_path',
                            message="Path to '.pth' weights file",
                            validate = lambda _, x: (os.path.isfile(x) and x.split(".")[-1] == 'pth'))
            ]
            weights = inq.prompt(question_weights_path)['weights_path']
    epoch_q = [
        inq.Text('num_epochs',
                    message="Set Number of Epochs to train for",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = 100),
    ]
    num_epochs = inq.prompt(epoch_q)['num_epochs']
    if not check_gpus():
        return
    train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs, weights)
    print ()
    print (f'{CLIColors.OKGREEN}Training finished! You KeypointDetect network is '
                f'ready for prediction, have fun :){CLIColors.ENDC}')
    print ()
    input("Press Enter to go back to main menu...")


def train_hybridnet():
    cls()
    print (f'{CLIColors.OKGREEN}Training HybridNet!{CLIColors.ENDC}')
    print ('You can change all network training setttings in the projects \'config.yaml\'.')
    print()
    projectManager = ProjectManager()
    projects = projectManager.get_projects()
    questions1 = [
        inq.List('project_name',
            message="Select project to load",
            choices=projects),
        inq.Text('weights_keypoint_detect',
                    message="Path to '.pth' KeypointDetect weights file",
                    validate = lambda _, x: ((os.path.isfile(x) and x.split(".")[-1] == 'pth') or x == "")),
    ]
    questions2 = [
        inq.Text('weights_hybridnet',
                    message="Path to '.pth' Hybridnet weights file",
                    validate = lambda _, x: ((os.path.isfile(x) and x.split(".")[-1] == 'pth') or x == ""))
    ]
    questions3 = [
        inq.Text('num_epochs',
                    message="Set Number of Epochs to train for",
                    validate = lambda _, x: (x.isdigit() and int(x) > 0),
                    default = 100),
        inq.List('mode',
            message="Select training mode",
            choices=['3D_only', 'last_layers', 'bifpn', 'all']),
    ]
    settings1 = inq.prompt(questions1)
    project_name = settings1['project_name']
    weights_keypoint_detect1 = settings1['weights_keypoint_detect']
    if weights_keypoint_detect1 == "":
        weights_keypoint_detect = None
        settings2 = inq.prompt(questions2)
        weights_hybridnet = settings2['weights_hybridnet']
        if weights_hybridnet == "":
            weights_hybridnet = None
    else:
        weights_hybridnet = None

    settings3 = inq.prompt(questions3)
    num_epochs = settings3['num_epochs']
    mode = settings3['mode']
    if mode == '3D_only':
        finetune = False
    else:
        finetune = True
    if not check_gpus():
        return
    train_interface.train_hybridnet(project_name, num_epochs,
                weights_keypoint_detect, weights_hybridnet, mode, finetune)
    print ()
    print (f'{CLIColors.OKGREEN}Training finished! You HybridNet is '
                f'ready for prediction, have fun :){CLIColors.ENDC}')
    print ()
    input("Press Enter to go back to main menu...")


def train_all():
    cls()
    print (f'{CLIColors.OKGREEN}Training Full Network Stack!{CLIColors.ENDC}')
    print ('You can change all network training setttings in the projects \'config.yaml\'.')
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
    ]
    settings = inq.prompt(questions)
    check_gpus()

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
            return
    print (f'Training KeypointDetect for {num_epochs_keypoint} epochs...')
    trained = train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs_keypoint, pretrain)
    if not trained:
        print ()
        input("Press Enter to go back to main menu...")
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
