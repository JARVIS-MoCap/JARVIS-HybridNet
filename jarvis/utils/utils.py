"""
utils.py
===============
Utility Funcions used across the whole HybridNet library
"""

import os
import json

class CLIColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_available_pretrains(parent_dir):
    pretrains = []
    subdirs = os.listdir(os.path.join(parent_dir, 'pretrained'))
    for dir in subdirs:
        files = os.listdir(os.path.join(parent_dir, 'pretrained', dir))
        if len(files) != 0:
            pretrains.append(dir)
            #TODO: Add more thorough checks here!
    return pretrains
