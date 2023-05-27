#!/usr/bin/env bash
#################################################
# Please do not make any changes to this file,  #
# change the variables in webui-user.sh instead #
#################################################

# change to local directory
cd -- "$(dirname -- "$0")"

can_run_as_root=0
export ERROR_REPORTING=FALSE
export PIP_IGNORE_INSTALLED=0

# Read variables from webui-user.sh
if [[ -f webui-user.sh ]]
then
    source ./webui-user.sh
fi

# python3 executable
if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

# git executable
if [[ -z "${GIT}" ]]
then
    export GIT="git"
fi

if [[ -z "${venv_dir}" ]]
then
    venv_dir="venv"
fi


# read any command line flags to the webui.sh script
while getopts "f" flag > /dev/null 2>&1
do
    case ${flag} in
        f) can_run_as_root=1;;
        *) break;;
    esac
done

# Do not run as root
if [[ $(id -u) -eq 0 && can_run_as_root -eq 0 ]]
then
    echo "Cannot run as root"
    exit 1
fi

for preq in "${GIT}" "${python_cmd}"
do
    if ! hash "${preq}" &>/dev/null
    then
        printf "Error: %s is not installed, aborting...\n" "${preq}"
        exit 1
    fi
done

if ! "${python_cmd}" -c "import venv" &>/dev/null
then
    echo "Error: python3-venv is not installed"
    exit 1
fi

echo "Create and activate python venv"
if [[ ! -d "${venv_dir}" ]]
then
    "${python_cmd}" -m venv "${venv_dir}"
    first_launch=1
fi

if [[ -f "${venv_dir}"/bin/activate ]]
then
    source "${venv_dir}"/bin/activate
else
    echo "Error: Cannot activate python venv"
    exit 1
fi

#Set OneAPI environmet if it's not set by the user
if [[ "$@" == *"--use-ipex"* ]] && ! [ -x "$(command -v sycl-ls)" ]
then
    echo "Setting OneAPI environment"
    if [[ -z "$ONEAPI_ROOT" ]]
    then
        ONEAPI_ROOT=/opt/intel/oneapi
    fi
    source $ONEAPI_ROOT/setvars.sh
fi

if [[ ! -z "${ACCELERATE}" ]] && [ ${ACCELERATE}="True" ] && [ -x "$(command -v accelerate)" ]
then
    echo "Launching accelerate launch.py..."
    exec accelerate launch --num_cpu_threads_per_process=6 launch.py "$@"
elif [[ -z "${first_launch}" ]] && [ -x "$(command -v ipexrun)" ] && [ -x "$(command -v numactl)" ] && [[ "$@" == *"--use-ipex"* ]]
then
    echo "Launching ipexrun launch.py..."
    exec ipexrun launch.py "$@"
else
    echo "Launching launch.py..."
    exec "${python_cmd}" launch.py "$@"
fi
