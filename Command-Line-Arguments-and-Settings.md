## webui-user
The recommended way to customize how the program is run is editing `webui-user.bat` (Windows) and `webui-user.sh` (Linux):
- `set PYTHON` allows for setting a custom Python path
    - Example: `set PYTHON=b:/soft/Python310/Python.exe`
- `set VENV_DIR` allows you to chooser the directory for the virtual environment. Default is `venv`. Special value `-` runs the script without creating virtual environment.
    - Example: `set VENV_DIR=C:\run\var\run` will create venv in the `C:\run\var\run` directory.
    - Example: `set VENV_DIR=-` runs the program using the system's python
- `set COMMANDLINE_ARGS` setting the command line arguments `webui.py` is ran with
    - Example: `set COMMANDLINE_ARGS=--ckpt a.ckpt` uses the model `a.ckpt` instead of `model.ckpt`

## Command Line Arguments
### Running online
Use the `--share` option to run online. You will get a xxx.app.gradio link. This is the intended way to use the program in collabs. You may set up authentication for said gradio shared instance with the flag `--gradio-auth username:password`, optionally providing multiple sets of usernames and passwords separated by commas.

Use `--listen` to make the server listen to network connections. This will allow computers on the local network to access the UI, and if you configure port forwarding, also computers on the internet.

Use `--port xxxx` to make the server listen on a specific port, xxxx being the wanted port. Remember that all ports below 1024 need root/admin rights, for this reason it is advised to use a port above 1024. Defaults to port 7860 if available.

# All command line arguments

| Argument Command | Value | Default | Description |
| ---------------- | ----- | ------- | ----------- |
| **CONFIGURATION** |
-h, --help         | None  | False   |   				  show this help message and exit |
--config    | CONFIG | configs/stable-diffusion/v1-inference.yaml   				 | path to config which constructs model |
--ckpt 		| CKPT   | model.ckpt        				 | path to checkpoint of stable diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded |
--ckpt-dir 	| CKPT_DIR | None   				 | Path to directory with stable diffusion checkpoints |
--gfpgan-dir| GFPGAN_DIR | GFPGAN/			 | GFPGAN directory |
--gfpgan-model| GFPGAN_MODEL			 | GFPGAN model file name |
--codeformer-models-path | CODEFORMER_MODELS_PATH | models/Codeformer/ | Path to directory with codeformer model file(s). |
--gfpgan-models-path | GFPGAN_MODELS_PATH | models/GFPGAN | Path to directory with GFPGAN model file(s). |
--esrgan-models-path | ESRGAN_MODELS_PATH | models/ESRGAN | Path to directory with ESRGAN model file(s). |
--bsrgan-models-path | BSRGAN_MODELS_PATH | models/BSRGAN | Path to directory with BSRGAN model file(s). |
--realesrgan-models-path | REALESRGAN_MODELS_PATH | models/RealESRGAN | Path to directory with RealESRGAN model file(s). |
--scunet-models-path | SCUNET_MODELS_PATH | models/ScuNET | Path to directory with ScuNET model file(s). |
--swinir-models-path | SWINIR_MODELS_PATH | models/SwinIR | Path to directory with SwinIR and SwinIR v2 model file(s). |
--ldsr-models-path | LDSR_MODELS_PATH | models/LDSR	| Path to directory with LDSR model file(s). |
--clip-models-path | CLIP_MODELS_PATH | None | Path to directory with CLIP model file(s). |
--vae-path | VAE_PATH | None  					| Path to Variational Autoencoders model |
--embeddings-dir | EMBEDDINGS_DIR | embeddings/		 | embeddings directory for textual inversion (default: embeddings) |
--hypernetwork-dir | HYPERNETWORK_DIR | models/hypernetworks/	 | hypernetwork directory |
--localizations-dir | LOCALIZATIONS_DIR | localizations/ | localizations directory
--styles-file | STYLES_FILE | styles.csv 				| filename to use for styles |
--ui-config-file | UI_CONFIG_FILE | 	ui-config.json	| filename to use for ui configuration |
--no-progressbar-hiding | None | False 			 | do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware acceleration in browser) |
--max-batch-count| MAX_BATCH_COUNT | 16	 | maximum batch count value for the UI |
--ui-settings-file | UI_SETTINGS_FILE | config.json 	| filename to use for ui settings |
--allow-code | None | False          				 | allow custom script execution from webui |
--share | None | False               				 | use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)
--listen  | None | False            					| launch gradio with 0.0.0.0 as server name, allowing to respond to network requests |
--port | PORT | 7860           					| launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available |
--hide-ui-dir-config | None | False  					| hide directory configuration from webui |
--freeze-settings | None | False | disable editing settings |
--enable-insecure-extension-access | None | False | enable extensions tab regardless of other options |
--gradio-debug | None | False        					| launch gradio with --debug option |
--gradio-auth | GRADIO_AUTH | None 				| set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3" |
--gradio-img2img-tool | {color-sketch,editor} | editor | gradio image uploader tool: can be either editor for ctopping, or color-sketch for drawing |
--disable-console-progressbars | None | False			| do not output progressbars to console |
--enable-console-prompts | None | False				| print prompts to console when generating with txt2img and img2img |
--api | None | False | launch webui with API |
--nowebui | None | False | only launch the API, without the UI |
--ui-debug-mode | None | Fales | Don't load model to quickly launch UI |
--device-id | DEVICE_ID | None | Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before) |
--administrator | None | False | Administrator rights |
| **PERFORMANCE** |
--xformers | None | False           					| enable xformers for cross attention layers |
--reinstall-xformers | None | False           					| force reinstall xformers. Useful for upgrading - but remove it after upgrading or you'll reinstall xformers perpetually. |
--force-enable-xformers	| None | False				| enable xformers for cross attention layers regardless of whether the checking code thinks you can run it; ***do not make bug reports if this fails to work*** |
--opt-split-attention | None | False 		| force-enables Doggettx's cross-attention layer optimization. By default, it's on for cuda enabled systems. |
--opt-split-attention-invokeai | None | False			| force-enables InvokeAI's cross-attention layer optimization. By default, it's on when cuda is unavailable. |
--opt-split-attention-v1 | None | False 				| enable older version of split attention optimization that does not consume all the VRAM it can find |
--opt-channelslast | None | False    					| change memory type for stable diffusion to channels last |
--disable-opt-split-attention | None | False 			| force-disables cross-attention layer optimization |
--use-cpu | {all, sd, interrogate, gfpgan, bsrgan, esrgan, scunet, codeformer} | None | use CPU as torch device for specified modules |
--no-half     | None | False         				 | do not switch the model to 16-bit floats |
--precision | {full,autocast} | autocast			 | evaluate at this precision |
--no-half-vae | None | False         				 | do not switch the VAE model to 16-bit floats |
--medvram    | None | False          				 | enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage |
--lowvram    | None | False          				 | enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage |
--lowram     | None | False         				 | load stable diffusion checkpoint weights to VRAM instead of RAM
--always-batch-cond-uncond | None | False			 | disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram
| **FEATURES** |
--autolaunch | None | False         					| open the webui URL in the system's default browser upon launch |
--theme | None | Unset         					| open the webui with the specified theme ("light" or "dark"). If not specified, uses the default browser theme |
--use-textbox-seed | None | False   					| use textbox for seeds in UI (no up/down, but possible to input long seeds) |
--disable-safe-unpickle | None | False				| disable checking pytorch models for malicious code |
--ngrok | NGROK | Unset         				 | ngrok authtoken, alternative to gradio --share
--ngrok-region | NGROK_REGION | Unset			 | The region in which ngrok should start.
--deepdanbooru | None | False       					| enable deepdanbooru interrogator |
| **DEFUNCT OPTIONS** |
--show-negative-prompt | None | False 					| does not do anything |
--unload-gfpgan | None | False      				 | does not do anything.