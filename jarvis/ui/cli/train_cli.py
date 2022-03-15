import click
import os
import torch

from jarvis.utils.utils import CLIColors
import jarvis.train_interface as train_interface


def set_gpu_environ(gpu):
    if gpu == None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)


@click.command(name='centerDetect')
@click.option('--num_epochs', default=None,
            type =click.IntRange(min=1),
            help='Number of Epochs, try 100 or more for very small datasets.')
@click.option('--weights_path', default=None,
            help='Weights to load before training. You have to specify the '
            'path to a specific \'.pth\' file')
@click.option('--pretrained_weights', default='None',
        type=click.Choice(['None', 'EcoSet', 'HumanHand', 'MonkeyHand',
        'HumanBody', 'RatBody', 'MouseBody'], case_sensitive=False),
        help='Pretrain to load before training. Select either \'EcoSet\' or '
        'one of the pretrained pose estimators available')
@click.option('--gpu', default=None,
            type=click.IntRange(min=0, max =torch.cuda.device_count()-1),
            help='Number of the GPU to be used')
@click.argument('project_name')
def train_center_detect(project_name, num_epochs, weights_path,
            pretrained_weights, gpu):
    set_gpu_environ(gpu)
    if weights_path != None:
        weights = weights_path
    elif pretrained_weights != 'None':
        weights = pretrained_weights
    else:
        weights = None
    train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs, weights)



@click.command(name='keypointDetect')
@click.option('--num_epochs', default=None,
            type =click.IntRange(min=1),
            help='Number of Epochs, try 200 or more for very small datasets.')
@click.option('--weights_path', default=None,
            help='Weights to load before training. You have to specify the '
            'path to a specific \'.pth\' file')
@click.option('--pretrained_weights', default='None',
            type=click.Choice(['None', 'EcoSet', 'HumanHand', 'MonkeyHand',
            'HumanBody', 'RatBody', 'MouseBody'], case_sensitive=False),
            help='Pretrain to load before training. Select either \'EcoSet\' or '
            'one of the pretrained pose estimators available')
@click.option('--gpu', default=None,
            type=click.IntRange(min=0, max =torch.cuda.device_count()-1),
            help='Number of the GPU to be used')
@click.argument('project_name')
def train_keypoint_detect(project_name, num_epochs, weights_path,
            pretrained_weights, gpu):
    set_gpu_environ(gpu)
    if weights_path != None:
        weights = weights_path
    elif pretrained_weights != 'None':
        weights = pretrained_weights
    else:
        weights = None
    train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs, weights)



@click.command(name = "hybridNet")
@click.option('--num_epochs', default=None, type=click.IntRange(min=1),
            help='Number of Epochs, try 100 or more for very small datasets.')
@click.option('--weights_hybridnet', default=None,
            help='Weights to load before training. You can specify the path to a'
                 ' specific \'.pth\' file, \'None\' for random initialization, '
                 '\'latest\' for most recently saved weights or \'ecoset\' for '
                 'EcoSet pretrained weights.')
@click.option('--weights_keypoint_detect', default='None',
            help='Weights to load for the 2D keypoint detect part of the network'
                 ' before training. You can specify the path to a'
                 ' specific \'.pth\' file or \'latest\' for most recently saved'
                 ' weights. Note that these'
                 ' weights will be overwritten if something else than \'None\''
                 ' is specified as \'weights\'')
@click.option('--mode', default='3D_only',
            type=click.Choice(['3D_only', 'last_layers', 'bifpn', 'all'],
            case_sensitive=False),
            help='Select which part of the network to train:\n'
            '\'all\': The whole network will be trained\n'
            '\'bifpn\': The whole network except the efficientnet backbone will'
            ' be trained\n'
            '\'last_layers\': The 3D network and the output layers of the 2D '
            'network will be trained\n'
            '\'3D_only\': Only the 3D network will be trained')
@click.option('--gpu', default=None,
            type=click.IntRange(min=0, max =torch.cuda.device_count()-1),
            help='Number of the GPU to be used')
@click.option('--finetune', default=False, type=click.BOOL,
            help='If True the whole network stack '
            'will be finetuned jointly. Might not fit into GPU Memory, '
            'depending on GPU model')
@click.argument('project_name')
def train_hybridnet(project_name, num_epochs, weights_keypoint_detect,
            weights_hybridnet, mode, gpu, finetune):
    """Train the full HybridNet on the project specified as PROJECT_NAME."""
    set_gpu_environ(gpu)
    if (weights_keypoint_detect == 'None'):
        weights_keypoint_detect = None
    if (weights_hybridnet == 'None'):
        weights_hybridnet = None
    train_interface.train_hybridnet(project_name, num_epochs,
                weights_keypoint_detect, weights_hybridnet, mode, finetune)



@click.command(name = "all")
@click.option('--num_epochs_center', default=None, type=click.IntRange(min=1),
            help='Number of Epochs for CenterDetect, try 100 or more for '
            'very small datasets.')
@click.option('--num_epochs_keypoint', default=None,type=click.IntRange(min=1),
            help='Number of Epochs for KeypointDetect, try 200 or more for '
            'very small datasets.')
@click.option('--num_epochs_hybridnet', default=None,type=click.IntRange(min=1),
            help='Number of Epochs for HybridNet, try 100 or more for '
            'very small datasets.')
@click.option('--pretrain', default='None',
            type=click.Choice(['None', 'HumanHand', 'MonkeyHand',
            'HumanBody', 'RatBody', 'MouseBody'], case_sensitive=False),
            help='Pretrain to load before training. Select '
            'one of the pretrained pose estimators available')
@click.option('--finetune', default=False, type=click.BOOL,
            help='If True the whole network stack will be finetuned jointly. '
            'Might not fit into GPU Memory, depending on GPU model')
@click.option('--gpu', default=None,
            type=click.IntRange(min=0, max =torch.cuda.device_count()-1),
            help='Number of the GPU to be used')
@click.argument('project_name')
def train_all(project_name, num_epochs_center, num_epochs_keypoint,
            num_epochs_hybridnet, pretrain, finetune, gpu):
    set_gpu_environ(gpu)
    click.echo(f'Training all newtorks for project {project_name} for '
                f'{num_epochs_center} epochs!')
    click.echo(f'First training CenterDetect for {num_epochs_center} epochs...')
    trained = train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs_center, pretrain)
    if not trained:
        return
    click.echo(f'Training KeypointDetect for {num_epochs_keypoint} epochs...')
    trained = train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs_keypoint, pretrain)
    if not trained:
        return
    click.echo(f'Training 3D section of HybridNet for {num_epochs_hybridnet} '
                f'epochs...')
    train_interface.train_hybridnet(project_name, num_epochs_hybridnet,
                'latest', None, '3D_only')
    if finetune:
        click.echo(f'Finetuning complete HybridNet for {num_epochs_hybridnet} '
                    f'epochs...')
        train_interface.train_hybridnet(project_name, num_epochs_hybridnet,
                    None, 'latest', 'all', finetune = True)
    click.echo()
    click.echo(f'{CLIColors.OKGREEN}Training finished! You networks are '
                f'ready for prediction, have fun :){CLIColors.ENDC}')