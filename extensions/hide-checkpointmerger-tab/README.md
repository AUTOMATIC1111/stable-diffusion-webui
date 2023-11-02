## HIDE CHECKPOINTMERGER TAB EXTENSION

# The one and only purpose of this extension is to disable completely the "Checkpoint Merger" tab in the StableDiffusion WebUI.

Disabling the "Checkpoint Merger" tab is possible in Settings -> User interface -> Hidden UI tabs
However, this extension overrides this setting (for example in order to prevent users to display the tab again)

--------------------

The script only comments one line, which can be found in the file modules/ui.py
This lines manages the display of the Checkpoint Merger tab, by commenting it, the WebUI can't display it.

Any changes can be reverted easily with the extension "revert-ui.py" or by downloading a fresh,
clean version of the module ui.py created by AUTOMATIC1111 (https://github.com/AUTOMATIC1111/stable-diffusion-webui)


## --------- AUTHOR: Olivier Gabelle ---------