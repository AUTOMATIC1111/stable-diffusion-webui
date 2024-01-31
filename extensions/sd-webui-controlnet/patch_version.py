import re
import subprocess


def get_current_version(filename):
    version_pattern = r"version_flag\s*=\s*'v(\d+\.\d+\.\d+)'"

    with open(filename, "r") as file:
        content = file.read()

    match = re.search(version_pattern, content)
    if match:
        return match.group(1)
    else:
        raise ValueError("Version number not found in the file")


def increment_version(version):
    major, minor, patch = map(int, version.split("."))
    patch += 1  # Increment the patch number
    return f"{major}.{minor}.{patch}"


def update_version_file(filename, new_version):
    with open(filename, "r") as file:
        content = file.read()

    new_content = re.sub(
        r"version_flag = 'v\d+\.\d+\.\d+'", f"version_flag = 'v{new_version}'", content
    )

    with open(filename, "w") as file:
        file.write(new_content)


def git_commit_and_tag(filename, new_version):
    commit_message = f":memo: Update to version v{new_version}"
    tag_name = f"v{new_version}"

    # Commit the changes
    subprocess.run(["git", "add", filename], check=True)
    subprocess.run(["git", "commit", "-m", commit_message], check=True)

    # Create a new tag
    subprocess.run(["git", "tag", tag_name], check=True)


if __name__ == "__main__":
    filename = "scripts/controlnet_version.py"
    current_version = get_current_version(filename)
    new_version = increment_version(current_version)
    update_version_file(filename, new_version)
    git_commit_and_tag(filename, new_version)
