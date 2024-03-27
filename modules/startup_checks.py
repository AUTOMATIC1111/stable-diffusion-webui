import sys
import os
import subprocess

import torch

## Functions for checking if user wants to skip startup checks
SKIP_STARTUP_CHECKS = "./skip_startup_checks.txt"

def startup_skip_check():
    if os.path.exists(SKIP_STARTUP_CHECKS):
        with open(SKIP_STARTUP_CHECKS, "r") as file:
            return file.readline()
    else:
        pass

## Functions for checks
# Performs gpu brand check
def get_gpu_brand():
    try:
        cuda_is_available = torch.cuda.is_available()
        if cuda_is_available == True:
            gpu_brand = torch.cuda.get_device_name()
            gpu_brand = gpu_brand.split()[0]
            print("\nGPU brand:", gpu_brand)
            return gpu_brand
        else:
            print("Cuda is not currently available")
    except:
        print("Torch or Cuda is not currently available")

# Performs vram check
def vram_check():
    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nUnable to get VRAM amount")

    if total_vram is not None:
        print(f"Total VRAM: {total_vram} MegaBytes")
        if total_vram > 8192:
            return 0
        elif total_vram <= 8192 and total_vram > 4096:
            return 1
        elif total_vram <= 4096:
            return 2
    else:
        return 3

# Function to write new args to webui-user.bat file
def write_commandline_args(new_commandline_args):
    bat_file = "./webui-user.bat"

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
    new_commandline_args = ""

    print("Running system checks")

    # Initial check to see if they wish to perform startup checks at each launch or not - It's really not necessary to
    skip_startup_checks = str(input("\nWould you like to skip these steps at each launch? (y)es or (n)o: ")).lower()
    while (skip_startup_checks != "yes") and (skip_startup_checks != "y") and (skip_startup_checks != "no") and (skip_startup_checks != "n"):
        print("\nPlease choose a valid response (y)es or (n)o")
        skip_startup_checks = str(input("\nWould you like to skip these steps at each launch? (y)es or (n)o: ")).lower()
    if (skip_startup_checks == "yes") or (skip_startup_checks == "y"):
        with open(SKIP_STARTUP_CHECKS, "w") as file:
            file.write("1")

    ## Skip webui install at each start?
    skip_webui_install = str(input("\nWould you like to skip stable-diffusion webui installation at each startup? (y)es or (n)o: ")).lower()
    while (skip_webui_install != "yes") and (skip_webui_install != "y") and (skip_webui_install != "no") and (skip_webui_install != "n"):
        print("\nPlease choose a valid response (y)es or (n)o")
        skip_webui_install = str(input("Would you like to skip stable-diffusion webui installation at each startup? (y)es or (n)o: ")).lower()
    if (skip_webui_install == "yes") or (skip_webui_install == "y"):
        new_commandline_args += "--skip-install "

    ## Perform checks and set variables
    # GPU brand
    gpu_brand = get_gpu_brand()
    if (gpu_brand == "NVIDIA"):
        new_commandline_args += "--xformers "

    # VRAM check
    vram_low_med = vram_check()

    if (vram_low_med == 0):
        pass
    elif (vram_low_med == 1):
        print("Medium vram")
        new_commandline_args += "--medvram "
    elif (vram_low_med == 2):
        print("Low vram")
        new_commandline_args += "--lowvram "
    else:
        pass

    # Ask user to confirm command line arguments, perform command line argument write
    print("\"", new_commandline_args, "\"")
    confirm_write = str(input("\nWould you like to set these arguments?\nPlease note that this will get rid of any existing arguments\n(y)es or (n)o: ")).lower()
    while (confirm_write != "yes") and (confirm_write != "y") and (confirm_write != "no") and (confirm_write != "n"):
        print("\nPlease choose a valid response (y)es or (n)o")
        confirm_write = str(input("Would you like to skip stable-diffusion webui installation at each startup? (y)es or (n)o: ")).lower()
    if (confirm_write == "yes") or (confirm_write == "y"):
        write_commandline_args(new_commandline_args)

# Checks if we should skip loading the startup checks
skip_startup_checks = startup_skip_check()
if skip_startup_checks == "1":
    pass
else:
    start_check_sequence()