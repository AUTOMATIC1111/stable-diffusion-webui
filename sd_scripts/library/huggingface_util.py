from typing import Union, BinaryIO
from huggingface_hub import HfApi
from pathlib import Path
import argparse
import os
from sd_scripts.library.utils import fire_in_thread


def exists_repo(repo_id: str, repo_type: str, revision: str = "main", token: str = None):
    api = HfApi(
        token=token,
    )
    try:
        api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type)
        return True
    except:
        return False


def upload(
    args: argparse.Namespace,
    src: Union[str, Path, bytes, BinaryIO],
    dest_suffix: str = "",
    force_sync_upload: bool = False,
):
    repo_id = args.huggingface_repo_id
    repo_type = args.huggingface_repo_type
    token = args.huggingface_token
    path_in_repo = args.huggingface_path_in_repo + dest_suffix if args.huggingface_path_in_repo is not None else None
    private = args.huggingface_repo_visibility is None or args.huggingface_repo_visibility != "public"
    api = HfApi(token=token)
    if not exists_repo(repo_id=repo_id, repo_type=repo_type, token=token):
        try:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private)
        except Exception as e:  # �Ȥꤢ����RepositoryNotFoundError�ϴ_�J���������ˤ��������Τ�
            print("===========================================")
            print(f"failed to create HuggingFace repo / HuggingFace: {e}")
            print("===========================================")

    is_folder = (type(src) == str and os.path.isdir(src)) or (isinstance(src, Path) and src.is_dir())

    def uploader():
        try:
            if is_folder:
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    folder_path=src,
                    path_in_repo=path_in_repo,
                )
            else:
                api.upload_file(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    path_or_fileobj=src,
                    path_in_repo=path_in_repo,
                )
        except Exception as e:  # RuntimeError��_�J�g�ߤ������ˤ��������Τ�
            print("===========================================")
            print(f"failed to upload to HuggingFace / HuggingFace�ؤΥ��åץ��`�ɤ�ʧ�����ޤ��� : {e}")
            print("===========================================")

    if args.async_upload and not force_sync_upload:
        fire_in_thread(uploader)
    else:
        uploader()


def list_dir(
    repo_id: str,
    subfolder: str,
    repo_type: str,
    revision: str = "main",
    token: str = None,
):
    api = HfApi(
        token=token,
    )
    repo_info = api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type)
    file_list = [file for file in repo_info.siblings if file.rfilename.startswith(subfolder)]
    return file_list
