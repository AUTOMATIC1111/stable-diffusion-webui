# Stable Diffusion web UI
A web interface for Stable Diffusion. Fork of A1111/stable-diffusion-web-ui. For Ubuntu with use of [PyEnv](https://github.com/pyenv/pyenv).

# Install
```
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc`
source ~/.bashrc
```

## Python Build Dependencies for PyEnv ([Wiki](https://github.com/pyenv/pyenv/wiki#suggested-build-environment))
### Mac
```
brew install openssl readline sqlite3 xz zlib tcl-tk
```

### Linux
```
sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## Python Venv
```
pyenv install 3.10.14
pyenv shell 3.10.14
```

# Start
Ensure there is a Python3.10 installation and it is pointed at in webui-user.sh.
```
sudo ./webui.sh
```
