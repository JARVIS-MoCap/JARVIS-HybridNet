from jarvis.utils.utils import CLIColors

def info(msg):
    print ("[Info] " + msg)

def warning(msg):
    print (f"{CLIColors.WARNING}[Warning] " + msg + f"{CLIColors.ENDC}")

def error(msg):
    print (f"{CLIColors.FAIL}[Error] " + msg + f"{CLIColors.ENDC}")

def success(msg):
    print (f"{CLIColors.OKGREEN}" + msg + f"{CLIColors.ENDC}")
