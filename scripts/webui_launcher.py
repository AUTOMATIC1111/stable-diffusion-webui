# This script only works on Windows!

import sys
import argparse
import os
import subprocess
import msvcrt
import configparser
import time
import signal

from turtle import clear


def main():
    parser = argparse.ArgumentParser()

    # These only really matter the first time you run the script, but they can be changed later in the config file
    # parser.add_argument("--git_path", default="git",
    #                     help="The path to the git executable (default: %(defaults)%)")
    # parser.add_argument("--git_branch", default="webui_bat_dev",  # remeber to change this
    #                     help="The git branch to use (default: %(defaults)%)")
    # parser.add_argument("--requirements_path", default="requirements_versions.txt",
    #                     help="The path to the requirements file to use (default: %(defaults)%)")

    # args = parser.parse_args()

    config = configparser.ConfigParser()

    launcher = Launcher(config)

    first_time = False
    if not os.path.exists("./config.ini") or os.path.exists("./tmp/first_time.txt"):
        # Create a temporary file to indicate that this is the first time the launcher is being run,
        # so if for whatever reason the launcher crashes or is closed, it will be able to detect that it is still the first time
        open("./tmp/first_time.txt", "w").close()

        config["LAUNCHER"] = {
            "git_branch": "",
            "git_path": "",
            "requirements_path": "",
            "commandline_args": "",
        }
        with open("./config.ini", "w") as configfile:
            config.write(configfile)
        first_time = True
    else:
        missing_section_exception = False
        try:
            config.read("./config.ini")
        except:
            missing_section_exception = True

        if "LAUNCHER" not in config or missing_section_exception:
            config["LAUNCHER"] = {
                "git_branch": "",
                "git_path": "",
                "requirements_path": "",
                "commandline_args": ""
            }
            with open("./config.ini", "w") as configfile:
                config.write(configfile)
            first_time = True

    if first_time:
        launcher.first_time_setup()

    launcher.check.model()

    while True:
        config.read("./config.ini")
        launcher.main_menu()


class Launcher:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.check = Check(self.config)
        self.installer = Installer(self.config, self.check)

    def main_menu(self):
        clear_screen()
        print(f"{bcolors.UNDERLINE}WebUI Launcher{bcolors.ENDC}")
        print(self.update_available(), end="")
        self.check.gfpgan()
        print("\nPress the number of the option you want to select.")
        print(f"{bcolors.BOLD}1 - Launch Webui")
        print("2 - Installer / Updater")
        print("3 - Options")
        print(f"4 - exit.{bcolors.ENDC}")
        print("\nWaiting for input...")

        key = wait_for_key(["1", "2", "3", "4"])

        match key:
            case "1": self.launch_webui()
            case "2": self.installer_menu()
            case "3": self.options_menu()
            case "4": self.exit()

    def first_time_setup(self):
        clear_screen()
        print(
            f"{bcolors.UNDERLINE}Do you wish to run the first time setup? (recommended){bcolors.ENDC}")
        print("Y/N?")

        match wait_for_key(["y", "n"]):
            case "y":
                clear_screen()
                print("Running first time setup...")
                self.first_time_config()
                self.installer.install_all()
                print("Finished setup. Press any key to continue...")
                wait_for_key()
                os.remove("./tmp/first_time.txt")
            case "n":
                os.remove("./tmp/first_time.txt")

    def first_time_config(self):
        """Gives the user the option to configure the git, branch and requirements file paths.
        """

        # Defaults
        git_path = "git"
        git_branch = "master"
        requirements_path = "requirements_versions.txt"

        clear_screen()

        print("Use the default launcher and installer settings?")
        print("Note: If you don't know what you're doing, it's recommended to leave the default settings.")
        print("(Y/N)")

        match wait_for_key(["y", "n"]):
            case "y":
                clear_screen()
                self.config.set("LAUNCHER", "git_path", git_path)
                self.config.set("LAUNCHER", "git_branch", git_branch)
                self.config.set("LAUNCHER", "requirements_path",
                                requirements_path)
            case "n":
                clear_screen()
                print("What is the path to the git executable? (default: git)")
                self.config.set("LAUNCHER", "git_path", input() or git_path)
                print(
                    f"\nWhat is the git branch you want to use? (default: {git_branch})")
                self.config.set("LAUNCHER", "git_branch",
                                input() or git_branch)
                print(
                    "\nWhat is the path to the requirements file? (default: requirements_versions.txt)")
                self.config.set("LAUNCHER", "requirements_path",
                                input() or requirements_path)

        with open("./config.ini", "w") as configfile:
            self.config.write(configfile)

    def launch_webui(self):
        clear_screen()
        try:
            subprocess.run([sys.executable, "webui.py"])
        except KeyboardInterrupt:
            return
        sys.exit()

    def installer_menu(self):
        clear_screen()
        print(f"{bcolors.UNDERLINE}Installer{bcolors.ENDC}")
        print("\nPress the number of the option you want to select.")
        print(f"{bcolors.BOLD}1 - Update and install")
        print("2 - Just install")
        print(f"3 - Go back{bcolors.ENDC}")
        print("\nWaiting for input...")

        key = wait_for_key(["1", "2", "3"])

        match key:
            case "1":
                clear_screen()
                self.installer.update()
                self.installer.install_all()
                print("Finished updating and installing. Press any key to continue...")
                wait_for_key()
            case "2":
                clear_screen()
                self.installer.install_all()
            case "3": return

        # subprocess.check_call([sys.executable, '-m', 'pip', '--help'])
        # print("Finished. Press any key to continue...")
        # wait_for_key()

    def options_menu(self):
        while True:
            clear_screen()
            print(f"{bcolors.UNDERLINE}Options{bcolors.ENDC}")
            print("\nPress the number of the option you want to select.")
            print(
                f"{bcolors.BOLD}1 - Change git path (current: {self.config.get('LAUNCHER', 'git_path')})")
            print(
                f"2 - Change git branch (current: {self.config.get('LAUNCHER', 'git_branch')})")
            print(
                f"3 - Change requirements path (current: {self.config.get('LAUNCHER', 'requirements_path')})")
            print(
                f"4 - Change commandline args (current: {self.config.get('LAUNCHER', 'commandline_args')})")
            print(f"5 - Edit config file directly")
            print(f"6 - Go back{bcolors.ENDC}")
            print("\nWaiting for input...")

            match wait_for_key(["1", "2", "3", "4", "5", "6"]):
                case "1":
                    clear_screen()
                    self.config.set("LAUNCHER", "git_path", input(
                        "Enter the path to the git executable (Leave empty for default): ") or "git")
                case "2":
                    clear_screen()
                    self.config.set("LAUNCHER", "git_branch", input(
                        "Enter the git branch you want to use (Leave empty for default): ") or "master")
                case "3":
                    clear_screen()
                    self.config.set("LAUNCHER", "requirements_path", input(
                        "Enter the path to the requirements file (Leave empty for default): ") or "requirements_versions.txt")
                case "4":
                    clear_screen()
                    self.config.set("LAUNCHER", "commandline_args", input(
                        "Enter the commandline args you want to use: ") or "")
                case "5":
                    clear_screen()
                    print(
                        "\nWhen you're done editing the config file, save, close and press any key to continue...")
                    os.startfile("config.ini")
                    wait_for_key()
                    self.config.read("./config.ini")
                case "6":
                    break

            with open("./config.ini", "w") as configfile:
                self.config.write(configfile)

    def exit(self):
        sys.exit()

    def update_available(self):
        new_commits = self.check.new_commits_count()
        return (f"{bcolors.OKGREEN}Update available! {new_commits} new commits. "
                f"Run the installer to update.{bcolors.ENDC}\n" if new_commits > 0 else "")


class Installer:
    def __init__(self, config: configparser.ConfigParser, check: "Check"):
        self.config = config
        self.check = check

    @ property
    def git_path(self):
        return self.config.get("LAUNCHER", "git_path")

    def _pip_install(self, package: str | list, *additional_args: str, from_file: bool = False):
        if isinstance(package, str):
            package = [package]
        try:
            # first check if the pip package is already installed
            if not from_file:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', *package, *additional_args])
            else:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-r', *package, *additional_args])
        except subprocess.CalledProcessError as e:
            print(f"{bcolors.FAIL}Error installing {package}{bcolors.ENDC}")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print("\nPress any key to quit...")
            wait_for_key()
            sys.exit(1)

    def _git_clone(self, repo: str, destination_dir: str):
        if "/" in destination_dir:
            destination_dir = destination_dir.replace("/", "\\")
        try:
            subprocess.check_call(
                [self.git_path, 'clone', repo, destination_dir])
        except subprocess.CalledProcessError as e:
            print(f"{bcolors.FAIL}Error cloning {repo}{bcolors.ENDC}")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print("\nPress any key to quit...")
            wait_for_key()
            sys.exit(1)

    def install_all(self):
        """Installs all required packages using pip
        """
        self.check.git()
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

            # Pip installs
            try:
                import torch  # type: ignore
            except ImportError:
                print("Installing torch...")
                self._pip_install(
                    "torch==1.12.1+cu113", "--extra-index-url", "https://download.pytorch.org/whl/cu113")

            self.check.gpu()

            try:
                import transformers  # type: ignore
                import wheel  # type: ignore
            except ImportError:
                print("Installing SD requirements...")
                self._pip_install(
                    ["wheel", "transformers==4.19.2",
                        "diffusers", "invisible-watermark"],
                    "--prefer-binary"
                )

            try:
                import k_diffusion.sampling  # type: ignore
            except ImportError:
                print("Installing K-Diffusion...")
                self._pip_install("git+https://github.com/crowsonkb/k-diffusion.git",
                                  "--prefer-binary", "--only-binary=psutil")

            try:
                import gfpgan  # type: ignore
            except ImportError:
                print("Installing GFPGAN...")
                self._pip_install("git+https://github.com/TencentARC/GFPGAN.git",
                                  "--prefer-binary")

            try:
                import omegaconf  # type: ignore
                import fonts  # type: ignore
                import timm  # type: ignore
            except ImportError:
                print("Installing requirements...")
                self._pip_install(self.config.get("LAUNCHER", "requirements_path"),
                                  "--prefer-binary", from_file=True)

            # create directory called "repositories" if it doesn't exist
            os.makedirs("repositoriess", exist_ok=True)

            # clone repositories
            if not os.path.exists("repositories/taming-transformers"):
                print("Cloning Taming Transforming repository...")
                self._git_clone("https://github.com/CompVis/taming-transformers.git",
                                "repositories/taming-transformers")

            if not os.path.exists("repositories/CodeFormer"):
                print("Cloning CLIP repository...")
                self._git_clone("https://github.com/sczhou/CodeFormer.git",
                                "repositories/CodeFormer")

            if not os.path.exists("repositories/BLIP"):
                print("Cloning BLIP repository...")
                self._git_clone("https://github.com/salesforce/BLIP.git",
                                "repositories/BLIP")
                try:
                    subprocess.check_call(
                        [self.git_path, "-C", "repositories/BLIP", "checkout", "48211a1594f1321b00f14c9f7a5b4813144b2fb9"])
                except subprocess.CalledProcessError as e:
                    print(f"{bcolors.FAIL}Error checking out BLIP{bcolors.ENDC}")
                    print(f"Command: {e.cmd}")
                    print(f"Return code: {e.returncode}")
                    print("\nPress any key to quit...")
                    wait_for_key()
                    sys.exit(1)

            try:
                import lpips  # type: ignore
            except ImportError:
                print("Installing requirements for CodeFormer...")
                self._pip_install("repositories/CodeFormer/requirements.txt",
                                  "--prefer-binary", from_file=True)

            print("\nFinished installing packages.")

        except subprocess.CalledProcessError as e:
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print("\nPress any key to quit...")
            wait_for_key()
            sys.exit(1)

    def update(self):
        # This currently assumes nothing weird has been done to the local git repo without
        # proper changes in config.ini
        self.check.git()
        branch = self.config.get("LAUNCHER", "git_branch")
        try:
            repo = subprocess.check_output(
                "git config --get remote.origin.url", shell=True).decode("utf-8").strip()
        except subprocess.CalledProcessError as e:
            print(f"{bcolors.FAIL}Error getting git repo{bcolors.ENDC}")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print("\nPress any key to skip...")
            wait_for_key()
            return

        print("Using remote repo: " + repo)
        print("Using branch: " + branch)
        print("Do you wish to continue? (Y/N)")

        match wait_for_key(["y", "n"]):
            case "y":
                pass
            case "n":
                print("Aborting update...")
                return

        try:
            subprocess.check_call(
                [self.git_path, "fetch", repo, branch], shell=True)

            subprocess.check_call(
                [self.git_path, "checkout", branch], shell=True)

            subprocess.check_call(
                [self.git_path, "pull"], shell=True)

        except subprocess.CalledProcessError as e:
            print("Update failed.")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print("\nPress any key to quit...")
            wait_for_key()
            sys.exit(1)


class Check:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config

    def model(self):
        """Checks if the model is placed in the root direcotry."""
        if not os.path.exists("model.ckpt"):
            print(f"Stable Diffusion model not found: you need to place model.ckpt file into same directory as this file.")
            print("\nDownload instructions:")
            print("1. Go to https://huggingface.co/CompVis/stable-diffusion-v1-4")
            print("2. log in or create an account")
            print("3. Click 'Access repository'.")
            print("4. Now download 'sd-v1-4.ckpt' from: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original")
            print(
                "5. Place it in the same directory as this file and Rename it to 'model.ckpt'.")
            print("\nPress R to retry, or ctrl+c to quit...")

            wait_for_key(["r"])
            self.model()

    def gfpgan(self):
        """Checks if the GFPGAN model is placed in the root direcotry."""
        if not os.path.exists("GFPGANv1.3.pth"):
            print(
                "\nGFPGAN not found: you need to place GFPGANv1.3.pth file into same directory as this file.")
            print("Face fixing feature will not work.")

    def git(self):
        """Checks if git is installed"""
        try:
            subprocess.check_output(
                [self.config.get('LAUNCHER', 'git_path'), "--version"])

        except FileNotFoundError as e:
            print(
                f"{bcolors.FAIL}Couldn't find git at path \"{self.config.get('LAUNCHER', 'git_path')}\".{bcolors.ENDC}")
            print("Please install git and try again: https://git-scm.com/downloads")
            print("\nPress any key to quit...")
            wait_for_key()
            sys.exit(1)

    def is_repo(self):
        try:
            subprocess.check_output(
                [self.config.get('LAUNCHER', 'git_path'), "rev-parse", "--is-inside-work-tree"], shell=True)
        except subprocess.CalledProcessError as e:
            print("This is not a git repository.")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print("\nPress any key to quit...")
            wait_for_key()
            sys.exit(1)

    def new_commits_count(self):
        """Returns the number of new commits since the last update"""
        try:
            new_commits = subprocess.check_output([self.config.get(
                "LAUNCHER", "git_path"), "rev-list", f"HEAD...origin/{self.config.get('LAUNCHER', 'git_branch')}", "--count"]).decode("utf-8").strip()
            return int(new_commits)
        except subprocess.CalledProcessError as e:
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print("\nPress any key to quit...")
            wait_for_key()
            sys.exit(1)

    def gpu(self) -> None:
        """Exits if cuda is not available"""
        import torch
        if torch.cuda.is_available():
            return
        else:
            print("Torch is not able to use GPU")
            print("Press any key to quit...")
            sys.exit(1)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def wait_for_key(keys: str | list[str] | None = None) -> str:
    """Wait for a specified key to be pressed.
    If no key is specified, wait for any key to be pressed.

    Args:
        keys: The key to wait for. If None, wait for any key.

    Returns:
        The key that was pressed.
    """

    if isinstance(keys, str):
        keys = [keys]

    # if there are strings that are not "enter", but are longer than 1 character, throw an error
    if keys is not None and any([len(x) > 1 and x != "enter" for x in keys]):
        raise ValueError("keys must be a single character or 'enter'")

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8")
            key = "enter" if key == "\r" else key
            if keys is None or key in keys:
                return key


def clear_screen():
    """Clear the console screen."""
    os.system("cls")


if __name__ == "__main__":
    main()
