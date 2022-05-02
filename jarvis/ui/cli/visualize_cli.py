import os
import click
from ruamel.yaml import YAML

from jarvis.utils.utils import CLIColors
import jarvis.utils.clp as clp
from jarvis.visualization.visualize_dataset import visualize_2D_sample, \
            visualize_3D_sample
from jarvis.config.project_manager import ProjectManager
from jarvis.visualization.create_videos3D import \
            create_videos3D as create_videos3D_func
from jarvis.visualization.create_videos2D import \
            create_videos2D as create_videos2D_func
from jarvis.utils.paramClasses import CreateVideos3DParams, CreateVideos2DParams

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


@click.command()
@click.option('--prediction_path', default = 'latest',
            help = 'Path to 3D prediction directory created by predict3D')
@click.option('--data_csv', default = 'data3D.csv',
            help = "Name of the prediction '.csv' file that is going to be "
            "used. If you ahve filtered results select the file accordingly.")
@click.argument('project_name')
def create_videos3D(project_name, prediction_path, data_csv):
    """
    Create videos overlayed with 3D poses for a multi-camera recording.
    """
    projectManager = ProjectManager()
    projectManager.load(project_name)
    cfg = projectManager.get_cfg()

    if prediction_path == 'latest':
        predict_root_path = os.path.join(projectManager.parent_dir,
                    cfg.PROJECTS_ROOT_PATH, project_name,
                    'predictions', 'predictions3D')
        if (os.path.exists(predict_root_path)
                    and len(os.listdir(predict_root_path)) != 0):
            dirs = os.listdir(predict_root_path)
            dirs = [os.path.join(predict_root_path, d) for d in dirs]
            dirs.sort(key=lambda x: os.path.getmtime(x))
            prediction_path = dirs[-1]
        else:
            clp.error("No Predictions found! Aborting...")
            return

    elif not os.path.exists(prediction_path):
        clp.error("Prediction Path does not exist! Aborting...")
        return

    if not os.path.exists(os.path.join(prediction_path, data_csv)):
        clp.error("DataCSV does not exist! Aborting...")
        return

    with open(os.path.join(prediction_path, 'info.yaml')) as file:
        yaml = YAML()
        info_yaml = yaml.load(file)
        recordings_path = info_yaml['recording_path']
        dataset_name = info_yaml['dataset_name']
        frame_start = info_yaml['frame_start']
        number_frames = info_yaml['number_frames']

    params = CreateVideos3DParams(project_name, recordings_path,
                os.path.join(prediction_path, data_csv))

    params.dataset_name = dataset_name
    params.frame_start = frame_start
    params.number_frames = number_frames

    cameras = []
    videos = os.listdir(params.recording_path)
    for video in os.listdir(params.recording_path):
        cameras.append(video.split('.')[0])
    params.video_cam_list = cameras

    create_videos3D_func(params)


@click.command()
@click.option('--prediction_path', default = 'latest',
            help = 'Path to 2D prediction directory created by predict3D')
@click.option('--data_csv', default = 'data2D.csv',
            help = "Name of the prediction '.csv' file that is going to be "
            "used. If you ahve filtered results select the file accordingly.")
@click.argument('project_name')
def create_videos2D(project_name, prediction_path, data_csv):
    """
    Create videos overlayed with 3D poses for a multi-camera recording.
    """
    projectManager = ProjectManager()
    projectManager.load(project_name)
    cfg = projectManager.get_cfg()

    if prediction_path == 'latest':
        predict_root_path = os.path.join(projectManager.parent_dir,
                    cfg.PROJECTS_ROOT_PATH, project_name,
                    'predictions', 'predictions2D')
        if (os.path.exists(predict_root_path)
                    and len(os.listdir(predict_root_path)) != 0):
            dirs = os.listdir(predict_root_path)
            dirs = [os.path.join(predict_root_path, d) for d in dirs]
            dirs.sort(key=lambda x: os.path.getmtime(x))
            prediction_path = dirs[-1]
        else:
            clp.error("No Predictions found! Aborting...")
            return

    elif not os.path.exists(prediction_path):
        clp.error("Prediction Path does not exist! Aborting...")
        return

    if not os.path.exists(os.path.join(prediction_path, data_csv)):
        clp.error("DataCSV does not exist! Aborting...")
        return

    with open(os.path.join(prediction_path, 'info.yaml')) as file:
        yaml = YAML()
        info_yaml = yaml.load(file)
        recording_path = info_yaml['recording_path']
        frame_start = info_yaml['frame_start']
        number_frames = info_yaml['number_frames']


    params = CreateVideos2DParams(project_name, recording_path,
                os.path.join(prediction_path, data_csv))

    params.frame_start = frame_start
    params.number_frames = number_frames

    create_videos2D_func(params)
