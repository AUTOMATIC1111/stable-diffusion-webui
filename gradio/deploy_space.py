from __future__ import annotations

import argparse
import os
import re

import huggingface_hub

import gradio as gr

repo_directory = os.getcwd()
readme_file = os.path.join(repo_directory, "README.md")
github_action_template = os.path.join(
    os.path.dirname(__file__), "deploy_space_action.yaml"
)


def add_configuration_to_readme(
    title: str | None,
    app_file: str | None,
) -> dict:
    configuration = {}

    dir_name = os.path.basename(repo_directory)
    if title is None:
        title = input(f"Enter Spaces app title [{dir_name}]: ") or dir_name
    formatted_title = format_title(title)
    if formatted_title != title:
        print(f"Formatted to {formatted_title}. ")
    configuration["title"] = formatted_title

    if app_file is None:
        for file in os.listdir(repo_directory):
            file_path = os.path.join(repo_directory, file)
            if not os.path.isfile(file_path) or not file.endswith(".py"):
                continue

            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "import gradio" in content:
                    app_file = file
                    break

        app_file = (
            input(f"Enter Gradio app file {f'[{app_file}]' if app_file else ''}: ")
            or app_file
        )
    if not app_file or not os.path.exists(app_file):
        raise FileNotFoundError("Failed to find Gradio app file.")
    configuration["app_file"] = app_file

    configuration["sdk"] = "gradio"
    configuration["sdk_version"] = gr.__version__
    huggingface_hub.metadata_save(readme_file, configuration)

    configuration["hardware"] = (
        input(
            f"Enter Spaces hardware ({', '.join(hardware.value for hardware in huggingface_hub.SpaceHardware)}) [cpu-basic]: "
        )
        or "cpu-basic"
    )

    secrets = {}
    if input("Any Spaces secrets (y/n) [n]: ") == "y":
        while True:
            secret_name = input("Enter secret name (leave blank to end): ")
            if not secret_name:
                break
            secret_value = input(f"Enter secret value for {secret_name}: ")
            secrets[secret_name] = secret_value
    configuration["secrets"] = secrets

    requirements_file = os.path.join(repo_directory, "requirements.txt")
    if (
        not os.path.exists(requirements_file)
        and input("Create requirements.txt file? (y/n) [n]: ").lower() == "y"
    ):
        while True:
            requirement = input("Enter a dependency (leave blank to end): ")
            if not requirement:
                break
            with open(requirements_file, "a") as f:
                f.write(requirement + "\n")

    if (
        input(
            "Create Github Action to automatically update Space on 'git push'? [n]: "
        ).lower()
        == "y"
    ):
        track_branch = input("Enter branch to track [main]: ") or "main"
        github_action_file = os.path.join(
            repo_directory, ".github/workflows/update_space.yml"
        )
        os.makedirs(os.path.dirname(github_action_file), exist_ok=True)
        with open(github_action_template) as f:
            github_action_content = f.read()
        github_action_content = github_action_content.replace("$branch", track_branch)
        with open(github_action_file, "w") as f:
            f.write(github_action_content)

        print(
            "Github Action created. Add your Hugging Face write token (from https://huggingface.co/settings/tokens) as an Actions Secret named 'hf_token' to your GitHub repository. This can be set in your repository's settings page."
        )

    return configuration


def format_title(title: str):
    title = title.replace(" ", "_")
    title = re.sub(r"[^a-zA-Z0-9\-._]", "", title)
    title = re.sub("-+", "-", title)
    while title.startswith("."):
        title = title[1:]
    return title


def deploy():
    if (
        os.getenv("SYSTEM") == "spaces"
    ):  # in case a repo with this function is uploaded to spaces
        return
    parser = argparse.ArgumentParser(description="Deploy to Spaces")
    parser.add_argument("deploy")
    parser.add_argument("--title", type=str, help="Spaces app title")
    parser.add_argument("--app-file", type=str, help="File containing the Gradio app")

    args = parser.parse_args()

    hf_api = huggingface_hub.HfApi()
    whoami = None
    login = False
    try:
        whoami = hf_api.whoami()
        if whoami["auth"]["accessToken"]["role"] != "write":
            login = True
    except OSError:
        login = True
    if login:
        print("Need 'write' access token to create a Spaces repo.")
        huggingface_hub.login(add_to_git_credential=False)
        whoami = hf_api.whoami()

    configuration: None | dict = None
    if os.path.exists(readme_file):
        try:
            configuration = huggingface_hub.metadata_load(readme_file)
        except ValueError:
            pass

    if configuration is None:
        print(
            f"Creating new Spaces Repo in '{repo_directory}'. Collecting metadata, press Enter to accept default value."
        )
        configuration = add_configuration_to_readme(
            args.title,
            args.app_file,
        )

    space_id = huggingface_hub.create_repo(
        configuration["title"],
        space_sdk="gradio",
        repo_type="space",
        exist_ok=True,
        space_hardware=configuration.get("hardware"),
    ).repo_id
    hf_api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=repo_directory,
    )
    if configuration.get("secrets"):
        for secret_name, secret_value in configuration["secrets"].items():
            huggingface_hub.add_space_secret(space_id, secret_name, secret_value)
    print(f"Space available at https://huggingface.co/spaces/{space_id}")
