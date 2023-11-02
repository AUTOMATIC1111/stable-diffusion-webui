## REVERT UI TAB EXTENSION

# The purpose of this extension is to revert the changes made by the following extensions : hide-checkpointmerger-tab, hide-apply-settings-button, hide-train-tab

Theses extensions are modifying lines in the modules/ui.py file of the StableDiffusion WebUI package.
The extension revert-ui.py is also modifying the file config.json in order to add itself to the list of disabled extensions by default.

--------------------

The script is designed to react to the changes made by the extensions listed previously, this may change in the future if extensions are added.

If anything is broken, you may want to download a fresh,
clean version of the  StableDiffusion WebUI package (created by AUTOMATIC1111 https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases)
to get a fresh version of ui.py. 
(the version of ui.py that I used to create my extensions can be found inside this extension as a backup)


## --------- AUTHOR: Olivier Gabelle ---------