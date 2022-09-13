#!/bin/bash
#################################################
# Please do not make any changes to this file,  #
# change the variables in webui-user.sh instead #
#################################################
# Read variables from webui-user.sh
# shellcheck source=/dev/null
if [[ -f webui-user.sh ]]
then
    source ./webui-user.sh
fi

# Set defaults
# Install directory without trailing slash
if [[ -z "${install_dir}" ]]
then
    install_dir="/home/$(whoami)"
fi

# Name of the subdirectory (defaults to stable-diffusion-webui)
if [[ -z "${clone_dir}" ]]
then
    clone_dir="stable-diffusion-webui"
fi

# Commandline arguments for webui.py, for example: commandline_args=(--medvram --opt-split-attention)
if [[ -z "${commandline_args}" ]]
then
    commandline_args=()
fi

# python3 executable
if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

# pip3 executable
if [[ -z "${pip_cmd}" ]]
then
    pip_cmd=(python3 -m pip)
fi

# git executable
if [[ -z "${git_cmd}" ]]
then
    git_cmd="git"
fi

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
if [[ -z "${venv_dir}" ]]
then
    venv_dir="venv"
fi

# pip3 install command for torch
if [[ -z "${torch_command}" ]]
then
    torch_command=(torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113)
fi

# Requirements file to use for stable-diffusion-webui
if [[ -z "${reqs_file}" ]]
then
    reqs_file="requirements_versions.txt"
fi

# Do not reinstall existing pip packages on Debian/Ubuntu
export PIP_IGNORE_INSTALLED=0

# Pretty print
delimiter="################################################################"

printf "\n%s\n" "${delimiter}"
printf "\e[1m\e[32mInstall script for stable-diffusion + Web UI\n"
printf "\e[1m\e[34mTested on Debian 11 (Bullseye)\e[0m"
printf "\n%s\n" "${delimiter}"

# Do not run as root
if [[ $(id -u) -eq 0 ]]
then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: This script must not be launched as root, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
else
    printf "\n%s\n" "${delimiter}"
    printf "Running on \e[1m\e[32m%s\e[0m user" "$(whoami)"
    printf "\n%s\n" "${delimiter}"
fi

if [[ -d .git ]]
then
    printf "\n%s\n" "${delimiter}"
    printf "Repo already cloned, using it as install directory"
    printf "\n%s\n" "${delimiter}"
    install_dir="${PWD}/../"
    clone_dir="${PWD##*/}"
fi

# Check prequisites
for preq in git python3
do
    if ! hash "${preq}" &>/dev/null
    then
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: %s is not installed, aborting...\e[0m" "${preq}"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi
done

if ! "${python_cmd}" -c "import venv" &>/dev/null
then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

printf "\n%s\n" "${delimiter}"
printf "Clone or update stable-diffusion-webui"
printf "\n%s\n" "${delimiter}"
cd "${install_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/, aborting...\e[0m" "${install_dir}"; exit 1; }
if [[ -d "${clone_dir}" ]]
then
    cd "${clone_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }
    "${git_cmd}" pull
else
    "${git_cmd}" clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "${clone_dir}"
    cd "${clone_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }
fi

printf "\n%s\n" "${delimiter}"
printf "Clone or update other repositories"
printf "\n%s\n" "${delimiter}"
if [[ ! -d repositories ]]
then
    mkdir repositories
fi
cd repositories || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/repositories/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }

for repo in stable-diffusion taming-transformers CodeFormer BLIP
do
    printf "\n%s\n" "${delimiter}"
    printf "%s" "${repo}"
    printf "\n%s\n" "${delimiter}"

    if [[ -d "${repo}" ]]
    then
        cd "${repo}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/stable-diffusion/repositories/%s, aborting...\e[0m" "${install_dir}" "${repo}"; exit 1; }
        "${git_cmd}" pull
        cd ..
    else
        if [[ "${repo}" == "stable-diffusion" || "${repo}" == "taming-transformers" ]]
        then
            "${git_cmd}" clone https://github.com/CompVis/"${repo}".git
        elif [[ "${repo}" == "CodeFormer" ]]
        then
            "${git_cmd}" clone https://github.com/sczhou/"${repo}".git
        elif [[ "${repo}" == "BLIP" ]]
        then
            "${git_cmd}" clone https://github.com/salesforce/"${repo}".git
        fi
    fi
done

printf "\n%s\n" "${delimiter}"
printf "Create and activate python venv"
printf "\n%s\n" "${delimiter}"
cd "${install_dir}"/"${clone_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }
if [[ ! -d "${venv_dir}" ]]
then
    "${python_cmd}" -m venv "${venv_dir}"
    first_launch=1
fi
# shellcheck source=/dev/null
if source "${venv_dir}"/bin/activate
then
    printf "\n%s\n" "${delimiter}"
    printf "Install dependencies"
    printf "\n%s\n" "${delimiter}"
    "${pip_cmd[@]}" install "${torch_command[@]}"
    "${pip_cmd[@]}" install wheel transformers==4.19.2 diffusers invisible-watermark --prefer-binary
    "${pip_cmd[@]}" install git+https://github.com/crowsonkb/k-diffusion.git@1a0703dfb7d24d8806267c3e7ccc4caf67fd1331 --prefer-binary --only-binary=psutil
    "${pip_cmd[@]}" install git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379 --prefer-binary
    "${pip_cmd[@]}" install -r "${reqs_file}" --prefer-binary
    "${pip_cmd[@]}" install -r repositories/CodeFormer/requirements.txt --prefer-binary
else
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

printf "\n%s\n" "${delimiter}"
printf "Check if models are present"
printf "\n%s\n" "${delimiter}"
for model in GFPGANv1.3.pth model.ckpt
do
    if [[ ! -f "${model}" ]]
    then
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[33mWarning:\e[0m %s file not found..." "${model}"
        printf "\n%s\n" "${delimiter}"
        if [[ "${model}" == "model.ckpt" ]] && [[ -n "${first_launch}" ]]
        then
            printf "\n%s\n" "${delimiter}"
            printf "Place \e[1m\e[32m%s\e[0m into webui directory, next to \e[1m\e[32mwebui.py\e[0m\n" "${model}"
            printf "Then press a key to continue...\n"
            read -rsn 1
            printf "\n%s\n" "${delimiter}"
        fi
    fi
done

printf "\n%s\n" "${delimiter}"
printf "Launching webui.py..."
printf "\n%s\n" "${delimiter}"
"${python_cmd}" webui.py "${commandline_args[@]}"
