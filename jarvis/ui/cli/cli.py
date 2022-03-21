import click
import os
import torch
from tensorboard import program

from jarvis.config.project_manager import ProjectManager
from jarvis.utils.utils import CLIColors



@click.command()
@click.option('--dataset2d', default = '',
            type = click.Path(exists = False, file_okay = False, dir_okay = True),
            help = 'Path to the dataset to be used for training the 2D parts of '
            'the network. Only specify if you have a pretraining dataset for '
            'the 2D network that differs from your 3D dataset.')
@click.option('--dataset3d', default = '',
            type = click.Path(exists = False, file_okay = False, dir_okay = True),
             help = 'Path to the dataset to be used for training the full '
             '3D network.')
@click.argument('project_name')
def create_project(project_name, dataset2d, dataset3d):
    """
    Create and configure a new project for your dataset.
    """
    if dataset3d == '' and dataset2d == '':
        print (f'{CLIColors.FAIL}Specify at least one dataset to create a '
                    'project. Aborting...{CLIColors.ENDC}')
        return
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
