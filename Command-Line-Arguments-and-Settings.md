## Environment variables

| name                   | description                                                                                                                               |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| PYTHON                 | sets a custom path for Python executable                                                                                                  |
| VENV_DIR               | specifies the path for the virtual environment. Default is `venv`. Special value `-` runs the script without creating virtual environment |
| COMMANDLINE_ARGS       | additional commandline arguments for the main program                                                                                     |
| IGNORE_CMD_ARGS_ERRORS | set to anything to make the program not exit with an error if an unedxpected commandline argument is encountered                          |
| REQS_FILE              | name of requirements.txt file with dependencies that wuill be installed when `launch.py` is run. Defaults to `requirements_versions.txt`  |
| TORCH_COMMAND          | command for installing pytorch                                                                                                            |
| INDEX_URL              | --index-url parameter for pip                                                                                                             |
| TRANSFORMERS_CACHE     | path to where transformers library will download and keep its files related to the CLIP model                                             |
| CUDA_VISIBLE_DEVICES   | select gpu to use for your instance on a system with multiple gpus. For example if you want to use secondary gpu, put "1".<br>(add a new line to webui-user.bat not in COMMANDLINE_ARGS): `set CUDA_VISIBLE_DEVICES=0`<br>Alternatively, just use `--device-id` flag in COMMANDLINE_ARGS. |

### webui-user
The recommended way to specify environment variables is by editing `webui-user.bat` (Windows) and `webui-user.sh` (Linux):
- `set VARNAME=VALUE` for Windows
- `export VARNAME="VALUE"` for Linux

For example, in Windows:
```
set COMMANDLINE_ARGS=--allow-code --xformers --skip-torch-cuda-test --no-half-vae --api --ckpt-dir A:\\stable-diffusion-checkpoints 
```

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
--exit | | | Terminate after installation |
--data-dir | DATA_DIR | ./ | base path where all user data is stored |
--config    | CONFIG | configs/stable-diffusion/v1-inference.yaml   				 | path to config which constructs model |
--ckpt 		| CKPT   | model.ckpt        				 | path to checkpoint of stable diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded |
--ckpt-dir 	| CKPT_DIR | None   				 | Path to directory with stable diffusion checkpoints |
--no-download-sd-model | None | False | don't download SD1.5 model even if no model is found |
--vae-dir | VAE_PATH | None  					| Path to Variational Autoencoders model | disables all settings related to VAE
--vae-path | VAE_PATH | None | Checkpoint to use as VAE; setting this argument
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
--lora-dir | LORA_DIR | models/Lora | Path to directory with Lora networks.
--clip-models-path | CLIP_MODELS_PATH | None | Path to directory with CLIP model file(s). |
--embeddings-dir | EMBEDDINGS_DIR | embeddings/		 | embeddings directory for textual inversion (default: embeddings) |
--textual-inversion-templates-dir | TEXTUAL_INVERSION_TEMPLATES_DIR | textual_inversion_templates | directory with textual inversion templates
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
--gradio-auth-path | GRADIO_AUTH_PATH | None | set gradio authentication file path ex. "/path/to/auth/file" same auth format as `--gradio-auth` |
--disable-console-progressbars | None | False			| do not output progressbars to console |
--enable-console-prompts | None | False				| print prompts to console when generating with txt2img and img2img |
--api | None | False | launch webui with API |
--api-auth | API_AUTH | None | Set authentication for API like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3" |
--api-log | None | False | enable logging of all API requests |
--nowebui | None | False | only launch the API, without the UI |
--ui-debug-mode | None | False | Don't load model to quickly launch UI |
--device-id | DEVICE_ID | None | Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before) |
--administrator | None | False | Administrator rights |
--cors-allow-origins | CORS_ALLOW_ORIGINS | None | Allowed CORS origin(s) in the form of a comma-separated list (no spaces) |
--cors-allow-origins-regex | CORS_ALLOW_ORIGINS_REGEX | None | Allowed CORS origin(s) in the form of a single regular expression |
--tls-keyfile | TLS_KEYFILE | None | Partially enables TLS, requires --tls-certfile to fully function |
--tls-certfile | TLS_CERTFILE | None | Partially enables TLS, requires --tls-keyfile to fully function |
--disable-tls-verify | None | False | When passed, enables the use of self-signed certificates.
--server-name | SERVER_NAME | None | Sets hostname of server |
--no-gradio-queue | None| False | Disables gradio queue; causes the webpage to use http requests instead of websockets; was the defaul in earlier versions
--no-hashing | None | False | disable sha256 hashing of checkpoints to help loading performance |
--skip-version-check | None | False | Do not check versions of torch and xformers |
--skip-python-version-check | None | False | Do not check versions of Python |
--skip-torch-cuda-test | None | False | do not check if CUDA is able to work properly |
--skip-install | None | False | skip installation of packages |
| **PERFORMANCE** |
--xformers | None | False           					| enable xformers for cross attention layers |
--force-enable-xformers	| None | False				| enable xformers for cross attention layers regardless of whether the checking code thinks you can run it; ***do not make bug reports if this fails to work*** |
--xformers-flash-attention | None | False | enable xformers with Flash Attention to improve reproducibility (supported for SD2.x or variant only)
--opt-sdp-attention | None | False | enable scaled dot product cross-attention layer optimization; requires PyTorch 2.*
--opt-sdp-no-mem-attention | False | None | enable scaled dot product cross-attention layer optimization without memory efficient attention, makes image generation deterministic; requires PyTorch 2.*
--opt-split-attention | None | False 		| force-enables Doggettx's cross-attention layer optimization. By default, it's on for cuda enabled systems. |
--opt-split-attention-invokeai | None | False			| force-enables InvokeAI's cross-attention layer optimization. By default, it's on when cuda is unavailable. |
--opt-split-attention-v1 | None | False 				| enable older version of split attention optimization that does not consume all the VRAM it can find |
--opt-sub-quad-attention | None | False | enable memory efficient sub-quadratic cross-attention layer optimization
--sub-quad-q-chunk-size | SUB_QUAD_Q_CHUNK_SIZE | 1024 | query chunk size for the sub-quadratic cross-attention layer optimization to use
--sub-quad-kv-chunk-size | SUB_QUAD_KV_CHUNK_SIZE | None | kv chunk size for the sub-quadratic cross-attention layer optimization to use
--sub-quad-chunk-threshold | SUB_QUAD_CHUNK_THRESHOLD | None | the percentage of VRAM threshold for the sub-quadratic cross-attention layer optimization to use chunking
--opt-channelslast | None | False    					| Enable alternative layout for 4d tensors, may result in faster inference **only** on Nvidia cards with Tensor cores (16xx and higher) |
--disable-opt-split-attention | None | False 			| force-disables cross-attention layer optimization |
--disable-nan-check | None | False | do not check if produced images/latent spaces have nans; useful for running without a checkpoint in CI
--use-cpu | {all, sd, interrogate, gfpgan, bsrgan, esrgan, scunet, codeformer} | None | use CPU as torch device for specified modules |
--no-half     | None | False         				 | do not switch the model to 16-bit floats |
--precision | {full,autocast} | autocast			 | evaluate at this precision |
--no-half-vae | None | False         				 | do not switch the VAE model to 16-bit floats |
--upcast-sampling | None | False | upcast sampling. No effect with --no-half. Usually produces similar results to --no-half with better performance while using less memory.
--medvram    | None | False          				 | enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage |
--lowvram    | None | False          				 | enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage |
--lowram     | None | False         				 | load stable diffusion checkpoint weights to VRAM instead of RAM
--always-batch-cond-uncond | None | False			 | disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram
| **FEATURES** |
--autolaunch | None | False         					| open the webui URL in the system's default browser upon launch |
--theme | None | Unset         					| open the webui with the specified theme ("light" or "dark"). If not specified, uses the default browser theme |
--use-textbox-seed | None | False   					| use textbox for seeds in UI (no up/down, but possible to input long seeds) |
--disable-safe-unpickle | None | False				| disable checking pytorch models for malicious code |
--ngrok | NGROK | None         				 | ngrok authtoken, alternative to gradio --share
--ngrok-region | NGROK_REGION | us			 | The region in which ngrok should start.
--update-check | None | None | On startup, notifies whether or not your webui version (commit) is up-to-date with che current master brach.
--update-all-extensions | None | None | On startup, it pulls the latest updates for all extensions you have installed.
--reinstall-xformers | None | False | force reinstall xformers. Useful for upgrading - but remove it after upgrading or you'll reinstall xformers perpetually. |
--reinstall-torch | None | False | force reinstall torch. Useful for upgrading - but remove it after upgrading or you'll reinstall torch perpetually. |
--tests | TESTS | False | Run test to validate webui functionality, see wiki topic for more details.
--no-tests | None | False | do not run tests even if --tests option is specified
| **DEFUNCT OPTIONS** |
--show-negative-prompt | None | False 					| does not do anything |
--deepdanbooru | None | False 					| does not do anything |
--unload-gfpgan | None | False      				 | does not do anything.
--gradio-img2img-tool | GRADIO_IMG2IMG_TOOL | None | does not do anything |
--gradio-inpaint-tool | GRADIO_INPAINT_TOOL | None | does not do anything |
--gradio-queue | None | False | does not do anything |
