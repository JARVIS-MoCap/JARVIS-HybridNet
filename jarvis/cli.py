import click
import os
import torch
from tensorboard import program
import streamlit.cli
import shutil
from pathlib import Path

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


@click.group()
def cli():
    pass


def set_gpu_environ(gpu):
    if gpu == None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)


@click.command()
def launch():
    project = ProjectManager()
    home = str(Path.home())
    os.makedirs(os.path.join(home, '.streamlit'), exist_ok = True)
    #stupid hack to stop streamlit (Version 1.5) from complaining
    info_name = click.get_current_context().parent.info_name
    click.get_current_context().parent.info_name = "streamlit"
    dirname = os.path.dirname(__file__)
    shutil.copyfile(os.path.join(project.parent_dir, 'jarvis/gui/config.toml'),
                os.path.join(home, '.streamlit', 'config.toml'))
    filename = os.path.join(dirname, 'gui', 'jarvis_gui.py')
    streamlit.cli._main_run(filename,
                flag_options={'primaryColor':'#2064a4', 'backgroundColor':'#222428'})
    click.get_current_context().parent.info_name = info_name



@click.command()
@click.option('--dataset2d', default='',
            help='Path to the dataset to be used for training the 2D parts of '
            'the network. Only specify if you have a pretraining dataset for '
            'the 2D network that differs from your 3D dataset.')
@click.option('--dataset3d', default='', help='Path to the dataset to be used '
            'for training the full 3D network.')
@click.argument('project_name')
def create_project(project_name, dataset2d, dataset3d):
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



@click.command()
@click.option('--num_epochs', default=None,
            help='Number of Epochs, try 100 or more for very small datasets.')
@click.option('--weights', default=None,
            help='Weights to load before training. You can specify the path to a'
                 ' specific \'.pth\' file, \'None\' for random initialization, '
                 '\'latest\' for most recently saved weights or \'ecoset\' for '
                 'EcoSet pretrained weights.')
@click.option('--gpu', default=None, help='Number of the GPU to be used')
@click.argument('project_name')
def train_center_detect(project_name, num_epochs, weights, gpu):
    set_gpu_environ(gpu)
    train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs, weights)



@click.command()
@click.option('--num_epochs', default=None,
            help='Number of Epochs, try 200 or more for very small datasets.')
@click.option('--weights', default=None,
            help='Weights to load before training. You can specify the path to a'
                 ' specific \'.pth\' file, \'None\' for random initialization, '
                 '\'latest\' for most recently saved weights or \'ecoset\' for '
                 'EcoSet pretrained weights.')
@click.option('--gpu', default=None, help='Number of the GPU to be used')
@click.argument('project_name')
def train_keypoint_detect(project_name, num_epochs, weights, gpu):
    set_gpu_environ(gpu)
    train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs, weights)



@click.command()
@click.option('--num_epochs', default=None,
            help='Number of Epochs, try 100 or more for very small datasets.')
@click.option('--weights', default=None,
            help='Weights to load before training. You can specify the path to a'
                 ' specific \'.pth\' file, \'None\' for random initialization, '
                 '\'latest\' for most recently saved weights or \'ecoset\' for '
                 'EcoSet pretrained weights.')
@click.option('--weights_keypoint_detect', default=None,
            help='Weights to load for the 2D keypoint detect part of the network'
                 ' before training. You can specify the path to a'
                 ' specific \'.pth\' file, \'latest\' for most recently saved'
                 ' weights or \'None\' for random initialization. Note that these'
                 ' weights will be overwritten if something else than \'None\''
                 ' is specified as \'weights\'')
@click.option('--mode', default='3D_only',
            help='Select which part of the network to train:\n'
            '\'all\': The whole network will be trained\n'
            '\'bifpn\': The whole network except the efficientnet backbone will'
            ' be trained\n'
            '\'last_layers\': The 3D network and the output layers of the 2D '
            'network will be trained\n'
            '\'3D_only\': Only the 3D network will be trained')
@click.option('--gpu', default=None, help='Number of the GPU to be used')
@click.option('--finetune', default=True, help='If True the whole network stack '
            'will be finetuned jointly. Might not fit into GPU Memory, '
            'depending on GPU model')
@click.argument('project_name')
def train_hybridnet(project_name, num_epochs, weights_keypoint_detect, weights,
            mode, gpu, finetune):
    """Train the full HybridNet on the project specified as PROJECT_NAME."""
    set_gpu_environ(gpu)
    train_interface.train_hybridnet(project_name, num_epochs,
                weights_keypoint_detect, weights, mode, finetune)



@click.command()
@click.option('--num_epochs_center', default=50, help='Number of Epochs for '
            'CenterDetect, try 100 or more for very small datasets.')
@click.option('--num_epochs_keypoint', default=100, help='Number of Epochs '
            'for KeypointDetect, try 200 or more for very small datasets.')
@click.option('--num_epochs_hybridnet', default=50, help='Number of Epochs '
            'for HybridNet, try 100 or more for very small datasets.')
@click.option('--finetune', default=True, help='If True the whole network '
            'stack will be finetuned jointly. Might not fit into GPU Memory, '
            'depending on GPU model')
@click.option('--gpu', default=None, help='Number of the GPU to be used')
@click.argument('project_name')
def train(project_name, num_epochs_center, num_epochs_keypoint,
            num_epochs_hybridnet, finetune, gpu):
    set_gpu_environ(gpu)
    click.echo(f'Training all newtorks for project {project_name} for '
                f'{num_epochs_center} epochs!')
    click.echo(f'First training CenterDetect for {num_epochs_center} epochs...')
    train_interface.train_efficienttrack('CenterDetect', project_name,
                num_epochs_center, 'ecoset')
    click.echo(f'Training KeypointDetect for {num_epochs_center} epochs...')
    train_interface.train_efficienttrack('KeypointDetect', project_name,
                num_epochs_keypoint, 'ecoset')
    click.echo(f'Training 3D section of HybridNet for {num_epochs_center} '
                f'epochs...')
    train_interface.train_hybridnet(project_name, num_epochs_hybridnet,
                'latest', None, '3D_only')
    if finetune:
        click.echo(f'Finetuning complete HybridNet for {num_epochs_center} '
                    f'epochs...')
        train_interface.train_hybridnet(project_name, num_epochs_hybridnet,
                    None, 'latest', 'all', finetune = True)
        click.echo()
        click.echo(f'{CLIColors.OKGREEN}Training finished! You networks are '
                    'ready for prediction, have fun :){CLIColors.ENDC}')



@click.command()
@click.option('--weights_center_detect', default='latest', help='TODO')
@click.option('--weights_keypoint_detect', default='latest', help='TODO')
@click.option('--output_dir', default=None, help='TODO')
@click.option('--frame_start', default=0, help='TODO')
@click.option('--number_frames', default=-1, help='TODO')
@click.option('--make_video', default=True, help='TODO')
@click.option('--skeleton_preset', default=None, help='TODO')
@click.argument('project_name')
@click.argument('video_path')
def predict_2D(project_name, video_path, weights_center_detect,
            weights_keypoint_detect, output_dir, frame_start, number_frames,
            make_video, skeleton_preset):
    predict_interface.predict2D(project_name, video_path, weights_center_detect,
                weights_keypoint_detect, output_dir, frame_start, number_frames,
                make_video, skeleton_preset)



@click.command()
@click.option('--weights_center_detect', default='latest', help='TODO')
@click.option('--weights_hybridnet', default='latest', help='TODO')
@click.option('--output_dir', default=None, help='TODO')
@click.option('--frame_start', default=0, help='TODO')
@click.option('--number_frames', default=-1, help='TODO')
@click.option('--make_videos', default=True, help='TODO')
@click.option('--skeleton_preset', default=None, help='TODO')
@click.option('--dataset_name', default=None, help='TODO')
@click.argument('project_name')
@click.argument('recording_path')
def predict_3D(project_name, recording_path, weights_center_detect,
            weights_hybridnet, output_dir, frame_start, number_frames,
            make_videos, skeleton_preset, dataset_name):
    predict_interface.predict3D(project_name, recording_path,
                weights_center_detect, weights_hybridnet, output_dir,
                frame_start, number_frames, make_videos, skeleton_preset,
                dataset_name)



@click.command()
@click.option('--start_frame', default = 0, help='TODO')
@click.option('--num_frames', default = 10, help='TODO')
@click.option('--skip_number', default=0, help='TODO')
@click.option('--skeleton_preset', default=None, help='TODO')
@click.option('--plot_azim', default=None, help='TODO')
@click.option('--plot_elev', default=None, help='TODO')
@click.argument('csv_file')
@click.argument('filename')
def plot_time_slices(csv_file, filename, start_frame, num_frames, skip_number,
            skeleton_preset, plot_azim,plot_elev):
    plot_slices(csv_file, filename,start_frame, num_frames, skip_number,
                skeleton_preset, plot_azim,plot_elev)



cli.add_command(create_project)
cli.add_command(train_center_detect)
cli.add_command(train_keypoint_detect)
cli.add_command(train_hybridnet)
cli.add_command(train)
cli.add_command(predict_2D)
cli.add_command(predict_3D)
cli.add_command(plot_time_slices)
cli.add_command(launch)
