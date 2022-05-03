"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v3.0
"""

import click
import os
import torch

from jarvis.utils.utils import CLIColors
import jarvis.train_interface as train_interface


@click.command(name='centerDetect')
@click.option('--num_epochs', default = None,
            type = click.IntRange(min = 1),
            help = 'Number of Epochs to run the training for, the default works '
            'well in almost all cases.')
@click.option('--weights_path', default = None,
            help = 'Weights to load before training. You have to specify the '
            'path to a specific \'.pth\' file')
@click.option('--pretrained_weights', default = 'None',
            help = "Pretrain to load before training. Select either 'EcoSet' "
            "or one of the pretrained pose estimators available")
@click.argument('project_name')
def train_center_detect(project_name, num_epochs, weights_path,
            pretrained_weights):
    """
    Train only the centerDetect network.
    """
    if weights_path != None:
        weights = weights_path
    elif pretrained_weights != 'None':
        weights = pretrained_weights
    else:
        weights = None
    train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs, weights)



@click.command(name='keypointDetect')
@click.option('--num_epochs', default = None,
            type = click.IntRange(min = 1),
            help = 'Number of Epochs to run the training for, the default works '
            'well in almost all cases.')
@click.option('--weights_path', default=None,
            help = 'Weights to load before training. You have to specify the '
            'path to a specific \'.pth\' file')
@click.option('--pretrained_weights', default='None',
            help = 'Pretrain to load before training. Select either \'EcoSet\' '
            'or one of the pretrained pose estimators available')
@click.argument('project_name')
def train_keypoint_detect(project_name, num_epochs, weights_path,
            pretrained_weights):
    """
    Train only the keypontDetect network.
    """
    if weights_path != None:
        weights = weights_path
    elif pretrained_weights != 'None':
        weights = pretrained_weights
    else:
        weights = None
    train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs, weights)



@click.command(name = "hybridNet")
@click.option('--num_epochs', default = None, type = click.IntRange(min = 1),
            help = 'Number of Epochs, the default works well in almost all cases.')
@click.option('--weights_hybridnet', default = None,
            help = 'Weights to load before training. You can specify the path '
            'to a specific \'.pth\' file, \'None\' for random initialization, '
            '\'latest\' for most recently saved weights or \'ecoset\' for '
            'EcoSet pretrained weights.')
@click.option('--weights_keypoint_detect', default = None,
            help = 'Weights to load for the 2D keypoint detect part of the '
            'network before training. You can specify the path to a'
            ' specific \'.pth\' file or \'latest\' for most recently saved'
            ' weights. Note that these'
            ' weights will be overwritten if something else than \'None\''
            ' is specified as \'weights_hybridnet\'')
@click.option('--mode', default = '3D_only',
            type = click.Choice(['3D_only', 'last_layers', 'bifpn', 'all'],
            case_sensitive = False),
            help = 'Select which part of the network to train:\n'
            '\'all\': The whole network will be trained\n'
            '\'bifpn\': The whole network except the efficientnet backbone will'
            ' be trained\n'
            '\'last_layers\': The 3D network and the output layers of the 2D '
            'network will be trained\n'
            '\'3D_only\': Only the 3D network will be trained')
@click.argument('project_name')
def train_hybridnet(project_name, num_epochs, weights_keypoint_detect,
            weights_hybridnet, mode):
    """
    Train the full HybridNet using previously trained keypointDetect weights.
    """
    if (weights_keypoint_detect == 'None'):
        weights_keypoint_detect = None
    if (weights_hybridnet == 'None'):
        weights_hybridnet = None
    if mode == '3D_only':
        finetune = False
    else:
        finetune = True
    train_interface.train_hybridnet(project_name, num_epochs,
                weights_keypoint_detect, weights_hybridnet, mode, finetune)



@click.command(name = "all")
@click.option('--num_epochs_center', default = None,
            type = click.IntRange(min = 1),
            help = 'Number of Epochs for CenterDetect.')
@click.option('--num_epochs_keypoint', default = None,
            type = click.IntRange(min = 1),
            help = 'Number of Epochs for KeypointDetect.')
@click.option('--num_epochs_hybridnet', default = None,
            type = click.IntRange(min = 1),
            help = 'Number of Epochs for HybridNet.')
@click.option('--pretrain', default='None',
            help = 'Pretrain to load before training. Select '
            'one of the pretrained pose estimators available')
@click.argument('project_name')
def train_all(project_name, num_epochs_center, num_epochs_keypoint,
            num_epochs_hybridnet, pretrain):
    """
    Train the full network stack from scratch.
    """
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
    click.echo()
    click.echo(f'{CLIColors.OKGREEN}Training finished! You networks are '
                f'ready for prediction, have fun :){CLIColors.ENDC}')
