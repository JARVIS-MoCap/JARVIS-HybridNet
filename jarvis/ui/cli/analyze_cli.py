import click
import os

import jarvis.analysis.analyze as analyze
import jarvis.analysis.plotting as plotting
from jarvis.config.project_manager import ProjectManager



def get_analysis_path(project_name):
    project = ProjectManager()
    project.load(project_name)
    cfg = project.get_cfg()
    analysis_path = os.path.join(project.parent_dir,
                project.cfg.PROJECTS_ROOT_PATH, project_name,
                'analysis')
    return analysis_path

@click.command()
@click.option('--weights_center_detect', default = 'latest',
            help = 'CenterDetect weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.option('--weights_hybridnet', default = 'latest',
            help = 'HybridNet weights to load for prediction. You have to '
            'specify the path to a specific \'.pth\' file')
@click.argument('project_name')
def analyze_validation_data(project_name, weights_center_detect,
            weights_hybridnet):
    """
    Analyse the validation data of your projects dataset.
    """
    analyze.analyze_validation_data(project_name, weights_center_detect,
                weights_hybridnet, None)




@click.command()
@click.option('--analysis_path', default = 'latest',
            help = 'Name of the directory containing the analysis csvs you '
            'want to use.')
@click.option('--cutoff', default = -1,
            help = 'Maximum error value to plot. Values bigger than the cutoff '
            'will be added to the last bin')
@click.argument('project_name')
def plot_error_histogram(project_name, analysis_path, cutoff):
    """
    Euclidean error across keypoints and time.
    """
    if analysis_path == 'latest':
        analysis_root_path = get_analysis_path(project_name)
        dirs = os.listdir(analysis_root_path)
        dirs = [os.path.join(analysis_root_path, d) for d in dirs] # add path to each file
        dirs.sort(key=lambda x: os.path.getmtime(x))
        analysis_path = dirs[-1]
    plotting.plot_error_histogram(analysis_path, {}, cutoff)


@click.command()
@click.option('--analysis_path', default = 'latest',
            help = 'Name of the directory containing the analysis csvs you '
            'want to use.')
@click.argument('project_name')
def plot_error_per_keypoint(project_name, analysis_path):
    """
    Euclidean error for each keypoint.
    """
    if analysis_path == 'latest':
        analysis_root_path = get_analysis_path(project_name)
        dirs = os.listdir(analysis_root_path)
        dirs = [os.path.join(analysis_root_path, d) for d in dirs] # add path to each file
        dirs.sort(key=lambda x: os.path.getmtime(x))
        analysis_path = dirs[-1]
    plotting.plot_error_per_keypoint(analysis_path)
