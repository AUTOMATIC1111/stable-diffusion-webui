## Quick Settings
<details><summary>The Quick Settings located at the top of the web page can be configured to your needs</summary>

`Setting User` -> `interface` -> `Quick settings list`
Any settings can be placed in the `Quick Settings`, changes to the settings hear will be immediately saved and applied and save to config.

![quick-settings-list-1](images/quick-settings-list-1.png)
![quick-settings-list-2](images/quick-settings-list-2.png)

By default we placed `Stable Diffusion checkpoint` in `Quick Settings`

Even though technically all settings can be relocated here, it doesn't make sense place settings that reloading or restarting to take effect in `Quick Settings`

</details>


## Additional Options for txt2img / img2img
How to get `Face Restore` or `Tiling` back
<details><summary>it's possible to add additional settings to txt2img or img2img</summary>

We allows user to add additional settings to the image generation interface, the settings can be found under

`Setting User` -> `interface` -> `Options in main UI - txt2img/img2img`
most if not all settings can be added here if needed

Previously `Face Restoration` and `Tiling` are built into the interface and cannot be modified, for users that finds them useful you can add them back manually

![additional-options-1](images/additional-options-1.png)
![additional-options-2](images/additional-options-2.png)

We also support additional option to change the look of how the options is displayed

</details>

## Gradio Themes
<details><summary>Customizing the basic look of webui</summary>

It is possible to customize the look of webui without using extensions suche as [Lobe Them](https://github.com/canisminor1990/sd-webui-lobe-theme.git) or [Nevysha's Cozy Nest](https://github.com/Nevysha/Cozy-Nest.git)

this can be done via gradio themes

![gradio-themes](images/gradio-themes.png)

We provide a small list of options choose from but you can manually input other themes from [gradio/theme-gallery](https://huggingface.co/spaces/gradio/theme-gallery)
if you find one you like you can inpot the corresponding `XXX/YYY` from the url `https://huggingface.co/spaces/XXX/YYY` in to ther dropdown menu

By default webui will cache the theme locally, this is so that it doesn't have to download it every time, but there's also means that if the theme is updated you won't received the updates to the theme, if you wish to update the theme (redownload) uncheck `Cache gradio themes locally` or or delete the corresponding theme cache.

The cached theme is stored under `stable-diffusion-webui/tmp/gradio_themes/your_selected_theme.json` (the slashes is replaced by underscore)

It also possible create your own theme locally or modify the cached themes

</details>