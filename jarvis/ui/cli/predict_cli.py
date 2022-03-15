import click
import jarvis.predict_interface as predict_interface


@click.command()
@click.option('--weights_center_detect', default='latest', help='TODO')
@click.option('--weights_keypoint_detect', default='latest', help='TODO')
@click.option('--frame_start', default=0, help='TODO')
@click.option('--number_frames', default=-1, help='TODO')
@click.option('--make_video', default=True, help='TODO')
@click.option('--skeleton_preset', default=None, help='TODO')
@click.argument('project_name')
@click.argument('video_path')
def predict_2D(project_name, video_path, weights_center_detect,
            weights_keypoint_detect, frame_start, number_frames,
            make_video, skeleton_preset):
    predict_interface.predict2D(project_name, video_path, weights_center_detect,
                weights_keypoint_detect, frame_start, number_frames,
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
