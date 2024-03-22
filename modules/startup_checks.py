import sys
import os
import subprocess
import json

## Functions for checking if user wants to skip startup
SKIP_STARTUP_CHECKS = "./skip_startup_checks.txt"

def startup_skip_check():
    if os.path.exists(SKIP_STARTUP_CHECKS):
        with open(SKIP_STARTUP_CHECKS, "r") as file:
            return file.readline()
    else:
        pass

## Functions for checks
def vram_check():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
        total_vram = sum(map(int, output.decode('utf-8').strip().split('\n')))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nUnable to get VRAM amount")

    if total_vram is not None:
        print(f"\nTotal VRAM available: {total_vram} bytes")
        if total_vram > 8192:
            return 0
        elif total_vram <= 8192 and total_vram > 4096:
            return 1
        elif total_vram <= 4096:
            return 2
    else:
        return 3

# Function to write new args to webui-user.bat file
def write_commandline_args():
    bat_file = "./webui-user.bat"
    ## TODO: Switch this to use actual dynamic commandline args from user options and system info
    new_commandline_args = ""

    # Read webui-user.bat file and save lines
    with open(bat_file, 'r') as file:
        lines = file.readlines()

    # Modify lines to have new commandline args
    for i, line in enumerate(lines):
        if line.strip().startswith('set COMMANDLINE_ARGS='):
            lines[i] = f'set COMMANDLINE_ARGS={new_commandline_args}\n'
            break

    # Write lines back to file
    with open(bat_file, 'w') as file:
        file.writelines(lines)

## Main starting check sequence
def start_check_sequence():
    print("Running system checks")

    # Initial check to see if they wish to perform startup checks at each launch or not - It's really not necessary to
    skip_startup_checks = str(input("\nWould you like to skip these steps at each launch? (y)es or (n)o: ")).lower()
    while (skip_startup_checks != "yes") and (skip_startup_checks != "y") and (skip_startup_checks != "no") and (skip_startup_checks != "n"):
        print("\nPlease choose a valid response (y)es or (n)o")
        skip_startup_checks = str(input("\nWould you like to skip these steps at each launch? (y)es or (n)o: ")).lower()
    if (skip_startup_checks == "yes") or (skip_startup_checks == "y"):
        with open(SKIP_STARTUP_CHECKS, "w") as file:
            file.write("1")

    # Skip webui install at each start?
    ## TODO: Implement check if already answered
    skip_webui_install = str(input("\nWould you like to skip stable-diffusion webui installation at each startup? (y)es or (n)o: ")).lower()
    while (skip_webui_install != "yes") and (skip_webui_install != "y") and (skip_webui_install != "no") and (skip_webui_install != "n"):
        print("\nPlease choose a valid response (y)es or (n)o")
        skip_webui_install = str(input("Would you like to skip stable-diffusion webui installation at each startup? (y)es or (n)o: ")).lower()
    if (skip_webui_install == "yes") or (skip_webui_install == "y"):
        ## TODO
        pass

    # Perform checks and set variables
    vram_low_med = vram_check()

    if vram_low_med == 0:
        pass
    elif vram_low_med == 1:
        # TODO
        print("Medium vram")
    elif vram_low_med == 2:
        # TODO
        print("Low vram")
    else:
        print("NVIDIA GPU or driver not found.")

# Checks if we should skip loading the startup checks
skip_startup_checks = startup_skip_check()
if skip_startup_checks == "1":
    pass
else:
    start_check_sequence()