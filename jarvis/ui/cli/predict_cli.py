import click
#import jarvis.predict_interface as predict_interface
from jarvis.prediction.predict3D import predict3D as predict3D_funct
from jarvis.utils.paramClasses import Predict3DParams


@click.command()
@click.option('--weights_center_detect', default = 'latest',
            help = 'CenterDetect weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--weights_keypoint_detect', default = 'latest',
            help = 'KeypointDetect weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--frame_start', default = 0,
            help = 'Number of the frame to start predictions at')
@click.option('--number_frames', default = -1,
            help = 'Number of frames to run predictions for. -1 to predict to '
            'the ende of the recording')
@click.option('--make_video', default = True,
            help = 'Select whether to make a video overlayed with the '
            'predictions')
@click.option('--skeleton_preset', default = "None",
            type = click.Choice(["None", "Hand", "HumanBody", "MonkeyBody",
            "RodentBody"], case_sensitive = False),
            help = 'Select a skelton preset that will be used for '
            'visualization.')
@click.argument('project_name')
@click.argument('video_path')
def predict2D(project_name, video_path, weights_center_detect,
            weights_keypoint_detect, frame_start, number_frames,
            make_video, skeleton_preset):
    """
    Predict 2D poses on a single video.
    """
    if skeleton_preset == "None":
        skeleton_preset = None
    predict_interface.predict2D(project_name, video_path, weights_center_detect,
                weights_keypoint_detect, frame_start, number_frames,
                make_video, skeleton_preset)


@click.command()
@click.option('--weights_center_detect', default = 'latest',
            help = 'CenterDetect weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--weights_hybridnet', default = 'latest',
            help = 'HybridNet weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--frame_start', default = 0,
            help = 'Number of the frame to start predictions at')
@click.option('--number_frames', default = -1,
            help = 'Number of frames to run predictions for. -1 to predict to '
            'the ende of the recording')
@click.option('--make_videos', default = True,
            help = 'Select whether to make a video overlayed with the '
            'predictions')
@click.option('--skeleton_preset', default = "None",
            type = click.Choice(["None", "Hand", "HumanBody", "MonkeyBody",
            "RodentBody"], case_sensitive = False),
            help = 'Select a skelton preset that will be used for '
            'visualization.')
@click.option('--dataset_name', default = None,
            help = 'If your dataset contains multiple calibrations, specify '
            'which one you want to use by giving the name of the dataset it '
            'belongs to. You can also specify a path to any folder containing '
            'valid calibrations for your recording.')
@click.argument('project_name')
@click.argument('recording_path')
def predict3D(project_name, recording_path, weights_center_detect,
            weights_hybridnet, frame_start, number_frames,
            make_videos, skeleton_preset, dataset_name):
    """
    Predict 3D poses on a multi-camera recording.
    """
    if skeleton_preset == "None":
        skeleton_preset = None

    params = Predict3DParams(project_name, recording_path)
    params.weights_center_detect = weights_center_detect
    params.weights_hybridnet = weights_hybridnet
    params.frame_start = frame_start
    params.number_frames = number_frames
    params.make_videos = make_videos
    params.skeleton_preset = skeleton_preset
    params.dataset_name = dataset_name
    predict3D_funct(params)
