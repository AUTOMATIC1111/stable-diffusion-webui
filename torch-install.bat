@echo off


start /wait cmd /k "%cd%\venv\Scripts\activate && pip install torch==2.1.0 torchvision==0.16.0 https://github.com/cavusmustafa/openvino/raw/custom_sdpa_optimizations/openvino-2023.2.0-12829-cp310-cp310-win_amd64.whl --force-reinstall && exit"

echo torch 2.1.0 dev installation completed.

powershell -executionpolicy bypass .\torch-install.ps1


echo eval_frame.py modification completed. press any key to exit
pause
