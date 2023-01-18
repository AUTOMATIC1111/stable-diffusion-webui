#/bin/env bash

python --version
pip3 --version
echo "installing general requirements"
pip3 install --disable-pip-version-check --quiet --requirement requirements.txt
echo "installing versioned requirements"
pip3 install --disable-pip-version-check --quiet --requirement requirements_versions.txt
