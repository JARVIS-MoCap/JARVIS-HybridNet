"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

from jarvis.utils.utils import CLIColors

def info(msg):
    print ("[Info] " + msg)

def warning(msg):
    print (f"{CLIColors.WARNING}[Warning] " + msg + f"{CLIColors.ENDC}")

def error(msg):
    print (f"{CLIColors.FAIL}[Error] " + msg + f"{CLIColors.ENDC}")

def success(msg):
    print (f"{CLIColors.OKGREEN}" + msg + f"{CLIColors.ENDC}")
