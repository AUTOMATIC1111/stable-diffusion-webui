import os
from helpers.commands import run



git = os.environ.get('GIT', "git")
repo_StableDiffusionWebUi = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui.git\n'


def get_remote_type(repoUrl):
    current_repo_origin = run(f"{git} remote get-url origin", None, "failed to check if it's a fork")
    is_fork = current_repo_origin != repoUrl
    return 'upstream' if is_fork else 'origin'

def get_diff_count(repoUrl):
    # function that gets the revision difference count between Remote Master and Local Master
    remote_type = get_remote_type(repoUrl)

    run(f"{git} fetch {remote_type}", None, f"failed to fetch from {remote_type}")

    #'{nBehind}\t{nAhead}\n'
    commit_diff_count = run(f"{git} rev-list --left-right --count {remote_type}/master...master") 
    commits_behind = int(commit_diff_count.split('\t')[0])

    # to fetch diff list
    # diff = run(f"{git} rev-list --left-right --pretty=oneline {remote_type}/master...master", None, f"failed to get diff list from {remote_type}")
    # diff_list = list(filter(None, diff.split('\n')))

    return commits_behind

def pull(repoUrl):
    # function that pulls the latest commits from remote
    remote_type = get_remote_type(repoUrl)

    merged_commits = run(f"{git} pull {remote_type} master:master", None, f"failed to merge from {remote_type} to master")
    print(f"Update Result:\n{merged_commits}")