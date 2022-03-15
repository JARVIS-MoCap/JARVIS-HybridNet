import click
import os
from pathlib import Path
import shutil
import streamlit.cli

from jarvis.config.project_manager import ProjectManager
import jarvis.ui.cli.cli as main_cli
import jarvis.ui.cli.train_cli as train_cli
import jarvis.ui.cli.predict_cli as predict_cli
import jarvis.ui.cli.visualize_cli as visualize_cli
import jarvis.ui.interactive_cli.cli as interactive_cli



@click.group()
def cli():
    pass

@cli.group()
def train():
    pass

@cli.group()
def predict():
    pass

@cli.group()
def visualize():
    pass


@click.command()
def hello():
    click.echo(f'{CLIColors.OKGREEN}Hi! JARVIS installed successfully and is '
                f'ready for training!\nRun \'jarvis --help\' to list all '
                f'available commands.{CLIColors.ENDC}')


@click.command()
def launch():
    project = ProjectManager()
    home = str(Path.home())
    os.makedirs(os.path.join(home, '.streamlit'), exist_ok = True)
    #stupid hack to stop streamlit (Version 1.5) from complaining
    info_name = click.get_current_context().parent.info_name
    click.get_current_context().parent.info_name = "streamlit"
    dirname = os.path.dirname(__file__)
    shutil.copyfile(os.path.join(project.parent_dir, 'jarvis/ui/gui/config.toml'),
                os.path.join(home, '.streamlit', 'config.toml'))
    filename = os.path.join(dirname, 'gui', 'jarvis_gui.py')
    streamlit.cli._main_run(filename,
                flag_options={'primaryColor':'#2064a4',
                'backgroundColor':'#222428'})
    click.get_current_context().parent.info_name = info_name


@click.command()
def launch_cli():
    interactive_cli.launch_interactive_prompt()


cli.add_command(hello)
cli.add_command(launch)
cli.add_command(launch_cli)
cli.add_command(main_cli.create_project)
train.add_command(train_cli.train_center_detect)
train.add_command(train_cli.train_keypoint_detect)
train.add_command(train_cli.train_hybridnet)
train.add_command(train_cli.train_all)
predict.add_command(predict_cli.predict_2D)
predict.add_command(predict_cli.predict_3D)
visualize.add_command(visualize_cli.plot_time_slices)
