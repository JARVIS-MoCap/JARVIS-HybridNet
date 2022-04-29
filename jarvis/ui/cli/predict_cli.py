import click
#import jarvis.predict_interface as predict_interface
from jarvis.prediction.predict3D import predict3D as predict3D_funct
from jarvis.prediction.predict2D import predict2D as predict2D_funct
from jarvis.utils.paramClasses import Predict3DParams
from jarvis.utils.paramClasses import Predict2DParams

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
@click.argument('project_name')
@click.argument('video_path')
def predict2D(project_name, video_path, weights_center_detect,
            weights_keypoint_detect, frame_start, number_frames):
    """
    Predict 2D poses on a single video.
    """
    params = Predict2DParams(project_name, video_path)
    params.weights_center_detect = weights_center_detect
    params.weights_keypoint_detect = weights_keypoint_detect
    params.frame_start = frame_start
    params.number_frames = number_frames
    predict2D_funct(params)


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
@click.option('--dataset_name', default = None,
            help = 'If your dataset contains multiple calibrations, specify '
            'which one you want to use by giving the name of the dataset it '
            'belongs to. You can also specify a path to any folder containing '
            'valid calibrations for your recording.')
@click.argument('project_name')
@click.argument('recording_path')
def predict3D(project_name, recording_path, weights_center_detect,
            weights_hybridnet, frame_start, number_frames, dataset_name):
    """
    Predict 3D poses on a multi-camera recording.
    """
    params = Predict3DParams(project_name, recording_path)
    params.weights_center_detect = weights_center_detect
    params.weights_hybridnet = weights_hybridnet
    params.frame_start = frame_start
    params.number_frames = number_frames
    params.dataset_name = dataset_name
    predict3D_funct(params)
