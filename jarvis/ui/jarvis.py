import click
import os
from pathlib import Path
import shutil
import collections
import streamlit.cli

from jarvis.config.project_manager import ProjectManager
import jarvis.ui.cli.cli as main_cli
import jarvis.ui.cli.train_cli as train_cli
import jarvis.ui.cli.predict_cli as predict_cli
import jarvis.ui.cli.visualize_cli as visualize_cli
import jarvis.ui.cli.analyze_cli as analyze_cli
import jarvis.ui.interactive_cli.cli as interactive_cli


class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands


@click.group(cls=OrderedGroup)
def cli():
    """
    Welcome to JARVIS! There are 3 ways to interact with the toolbox:\n
      1. The standard CLI, see this help for all available commands\n
      2. The interactive CLI: run 'jarvis launch-cli' to open it here\n
      3. The streamlit GUI: run 'jarvis launch' to open it in your browser
    """
    pass



@click.command()
def launch():
    """
    Launch the Streamlit GUI in your browser.
    """
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
    """
    Launch the interactive CLI in this terminal.
    """
    interactive_cli.launch_interactive_prompt()


cli.add_command(launch)
cli.add_command(launch_cli)
cli.add_command(main_cli.create_project)

@cli.group()
def train():
    """
    Training commands, more info: 'jarvis train --help'.
    """
    pass

@cli.group()
def predict():
    """
    Prediction commands, more info: 'jarvis predict --help'.
    """
    pass

@cli.group()
def visualize():
    """
    Visualize commands, more info: 'jarvis visualize --help'.
    """
    pass

@cli.group()
def analyze():
    """
    Analysis commands, more info: 'jarvis analyze --help'.
    """
    pass

train.add_command(train_cli.train_center_detect)
train.add_command(train_cli.train_keypoint_detect)
train.add_command(train_cli.train_hybridnet)
train.add_command(train_cli.train_all)
predict.add_command(predict_cli.predict2D)
predict.add_command(predict_cli.predict3D)
visualize.add_command(visualize_cli.create_videos3D)
visualize.add_command(visualize_cli.create_videos2D)
analyze.add_command(analyze_cli.analyze_validation_data)
analyze.add_command(analyze_cli.plot_error_histogram)
analyze.add_command(analyze_cli.plot_error_per_keypoint)
analyze.add_command(analyze_cli.plot_error_histogram_per_keypoint)
