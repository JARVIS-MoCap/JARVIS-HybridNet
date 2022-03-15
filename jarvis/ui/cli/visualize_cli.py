import click

from jarvis.utils.utils import CLIColors
import jarvis.visualize_interface as visualize

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
