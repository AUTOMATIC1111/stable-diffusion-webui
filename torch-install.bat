@echo off


start /wait cmd /k "%cd%\venv\Scripts\activate && pip install %cd%\torch-2.1.0.dev20230713+cpu-cp310-cp310-win_amd64.whl && exit"

echo torch 2.1.0 dev installation completed.

powershell -executionpolicy bypass .\torch-install.ps1


echo eval_frame.py modification completed. press any key to exit
pause
