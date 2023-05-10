@echo off

set PYTHON=C:\ProgramData\Anaconda3\envs\image_generation\python.exe
:: it must be python 3.10 to work properly
::set PYTHON=C:\Users\Animefest\miniconda3\python.exe
set GIT=
set VENV_DIR=
:: the hide-ui-dir-config is separate flag, hides server-side
::set COMMANDLINE_ARGS=--theme dark --ui-settings-file=config-af.json --ui-config-file=ui-config-af.json --hide-ui-dir-config --freeze-settings
:: share to local network by adding --share --listen
::set COMMANDLINE_ARGS=--theme dark --ui-settings-file=config-af.json --ui-config-file=ui-config-af.json --hide-ui-dir-config --freeze-settings --share --listen
:: same settings for some stuff, but unlocked UI
::set COMMANDLINE_ARGS=--theme dark --ui-settings-file=config-af.json --ui-config-file=ui-config-default.json --hide-ui-dir-config --freeze-settings --share --listen
::unlocked version of AF, for local network and allowing to install stuff, very dangerous, do not run publicly
set COMMANDLINE_ARGS=--theme dark --ui-settings-file=config-af-unlocked.json --ui-config-file=ui-config-af-unlocked.json --hide-ui-dir-config --share --listen --enable-insecure-extension-access
:: for my laptop
::set COMMANDLINE_ARGS=--theme dark --ui-settings-file=config-af.json --hide-ui-dir-config --ui-config-file=ui-config-ntb.json
:: for the full-config in my fork
::set COMMANDLINE_ARGS=--theme dark --ui-settings-file=config.json --ui-config-file=ui-config-default.json
::set COMMANDLINE_ARGS=--theme dark --medvram
::set COMMANDLINE_ARGS=--theme dark --lowvram
::set COMMANDLINE_ARGS=--theme dark --use-cpu all --no-half


call webui.bat
