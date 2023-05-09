- **The program is tested to work on Python 3.10.6. Don't use other versions unless you are looking for trouble.**
- The program needs 16gb of regular RAM to run smoothly. If you have 8gb RAM, consider making an 8gb page file/swap file, or use the [--lowram](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings) option (if you have more gpu vram than ram).
- The installer creates a python virtual environment, so none of the installed modules will affect existing system installations of python.
- To use the system's python rather than creating a virtual environment, use custom parameter replacing `set VENV_DIR=-`.
- To reinstall from scratch, delete directories: `venv`, `repositories`.
- When starting the program for the first time, the path to python interpreter is displayed. If this is not the python you installed, you can specify full path in the `webui-user` script; see [Running with custom parameters](Run-with-Custom-Parameters).
- If the desired version of Python is not in PATH, modify the line `set PYTHON=python` in `webui-user.bat` with the full path to the python executable.
    - Example: `set PYTHON=B:\soft\Python310\python.exe`
- Installer requirements from `requirements_versions.txt`, which lists versions for modules specifically compatible with Python 3.10.6. If this doesn't work with other versions of Python, setting the custom parameter `set REQS_FILE=requirements.txt` may help.

# Low VRAM Video-cards
When running on video cards with a low amount of VRAM (<=4GB), out of memory errors may arise.
Various optimizations may be enabled through command line arguments, sacrificing some/a lot of speed in favor of using less VRAM:
- If you have 4GB VRAM and want to make 512x512 (or maybe up to 640x640) images, use `--medvram`.
- If you have 4GB VRAM and want to make 512x512 images, but you get an out of memory error with `--medvram`, use `--lowvram --always-batch-cond-uncond` instead.
- If you have 4GB VRAM and want to make images larger than you can with `--medvram`, use  `--lowvram`.

# Torch is not able to use GPU
This is one of the most frequently mentioned problems, but it's usually not a WebUI fault, there are many reasons for it.
- WebUI uses GPU by default, so if you don't have suitable hardware, you need to add `--use-cpu`.
- Make sure you configure the WebUI correctly, refer to the corresponding installation tutorial in the [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).
- If you encounter this issue after some component updates, try undoing the most recent actions.

If you are one of the above, you should delete the `venv` folder.

If you still can't solve the problem, you need to submit some additional information when reporting.
1. Open the console under `venv\Scripts`
2. Run `python -m torch.utils.collect_env`
3. Copy all the output of the console and post it

# Green or Black screen
Video cards
Certain GPU video cards don't support half precision: a green or black screen may appear instead of the generated pictures. Use `--upcast-sampling`. This should stack with `--xformers` if you are using.
If still not fixed, use command line arguments `--precision full --no-half` at a significant increase in VRAM usage, which may require `--medvram`.

# "CUDA error: no kernel image is available for execution on the device" after enabling xformers
Your installed xformers is incompatible with your GPU. If you use Python 3.10, have a Pascal or higher card and run on Windows, add `--reinstall-xformers --xformers` to your `COMMANDLINE_ARGS` to upgrade to a working version. Remove `--reinstall-xformers` after upgrading.

# NameError: name 'xformers' is not defined
If you use Windows, this means your Python is too old. Use 3.10

If Linux, you'll have to build xformers yourself or just avoid using xformers.

# `--share` non-functional after gradio 3.22 update

Windows defender/antiviruses sometimes blocks Gradio's ability to create a public URL.

1. Go to your antivirus
2. Check the protection history: \
![image](https://user-images.githubusercontent.com/98228077/229028161-4ad3c837-ae3f-45f7-9a0a-fa165d70d943.png)
3. Add it as an exclusion

Related issues:
<details>

https://github.com/gradio-app/gradio/issues/3230 \
https://github.com/gradio-app/gradio/issues/3677
</details>

# weird css loading

![image](https://user-images.githubusercontent.com/98228077/229085355-0fbd56d6-fe1c-4858-8701-6c5697b9a6d6.png)

This issue has been noted, 3 times. It is apparently something users in china may experience.
[#8537](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8537)

<details><summary> Solution: </summary>

This problem is caused by errors in the CSS file type information in my computer registry, which leads to errors in CSS parsing and application.
Solution:

![image](https://user-images.githubusercontent.com/98228077/229086022-f27858a3-c9d9-470c-87cc-aa1974b7c5d0.png)


According to the above image to locate, and modify the last Content Type and PerceivedType.
Finally, reboot the machine, delete the browser cache, and force refresh the web page (shift+f5).
Thanks to https://www.bilibili.com/read/cv19519519
</details>