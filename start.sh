USERNAME=novelai

if id -u ${USERNAME} >/dev/null 2>&1 ; then
    echo "User ${USERNAME} exists."
else
    echo "Creating ${USERNAME}."
    useradd ${USERNAME}
    passwd ${USERNAME} ${USERNAME}
    mkdir /home/${USERNAME}
fi

su ${USERNAME} - -c ./webui.sh