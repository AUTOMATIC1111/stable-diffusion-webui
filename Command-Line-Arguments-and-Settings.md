## Environment variables

| Name                   | Description                                                                                                                               |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| PYTHON                 | Sets a custom path for Python executable.                                                                                                  |
| VENV_DIR               | Specifies the path for the virtual environment. Default is `venv`. Special value `-` runs the script without creating virtual environment. |
| COMMANDLINE_ARGS       | Additional commandline arguments for the main program.                                                                                     |
| IGNORE_CMD_ARGS_ERRORS | Set to anything to make the program not exit with an error if an unexpected commandline argument is encountered.                          |
| REQS_FILE              | Name of `requirements.txt` file with dependencies that will be installed when `launch.py` is run. Defaults to `requirements_versions.txt`.  |
| TORCH_COMMAND          | Command for installing PyTorch.                                                                                                            |
| INDEX_URL              | `--index-url` parameter for pip.                                                                                                             |
| TRANSFORMERS_CACHE     | Path to where transformers library will download and keep its files related to the CLIP model.                                             |
| CUDA_VISIBLE_DEVICES   | Select GPU to use for your instance on a system with multiple GPUs. For example, if you want to use secondary GPU, put "1".<br>(add a new line to webui-user.bat not in COMMANDLINE_ARGS): `set CUDA_VISIBLE_DEVICES=0`<br>Alternatively, just use `--device-id` flag in `COMMANDLINE_ARGS`. |
| SD_WEBUI_LOG_LEVEL   | Log verbosity. Supports any valid logging level supported by Python's built-in `logging` module. Defaults to `INFO` if not set. |
| SD_WEBUI_CACHE_FILE   | Cache file path. Defaults to `cache.json` in the root directory if not set. |
| SD_WEBUI_RESTAR | A value set by `launcher script` (like webui.bat webui.sh) informing Webui that restart function is available |
| SD_WEBUI_RESTARTING | A internal value signifying if webui is currently restarting or reloading, used for disabling certain actions asuch as auto launching browser.<br>set to `1` disables auto launching browser<br>set to `0` enables auto launch even when restarting<br>Certain extensions might use this value for similar purpose. |

### webui-user
The recommended way to specify environment variables is by editing `webui-user.bat` (Windows) and `webui-user.sh` (Linux):
- `set VARNAME=VALUE` for Windows
- `export VARNAME="VALUE"` for Linux

For example, in Windows:
```
set COMMANDLINE_ARGS=--allow-code --xformers --skip-torch-cuda-test --no-half-vae --api --ckpt-dir A:\\stable-diffusion-checkpoints 
```

### Running online
Use the `--share` option to run online. You will get a xxx.app.gradio link. This is the intended way to use the program in colabs. You may set up authentication for said gradio shared instance with the flag `--gradio-auth username:password`, optionally providing multiple sets of usernames and passwords separated by commas.

### Running within Local Area Network
Use `--listen` to make the server listen to network connections. This will allow computers on the local network to access the UI, and if you configure port forwarding, also computers on the internet. Example address: `http://192.168.1.3:7860`
Where your "192.168.1.3" is the local IP address.

Use `--port xxxx` to make the server listen on a specific port, xxxx being the wanted port. Remember that all ports below 1024 need root/admin rights, for this reason it is advised to use a port above 1024. Defaults to port 7860 if available.

### Running on CPU
Running with only your CPU is possible, but not recommended. 
It is very slow and there is no fp16 implementation. 

To run, you must have all these flags enabled: `--use-cpu all --precision full --no-half --skip-torch-cuda-test`

Though this is a questionable way to run webui, due to the very slow generation speeds; using the various AI upscalers and captioning tools may be useful to some people.

<details><summary>Extras: </summary>

For the technically inclined, here are some steps a user provided to boost CPU performance:

https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10514

https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10516

</details>

# All command line arguments

| Argument Command | Value | Default | Description |
| ---------------- | ----- | ------- | ----------- |
| **CONFIGURATION** |
-h, --help         | None  | False   |   				  Show this help message and exit. |
--exit | | | Terminate after installation |
--data-dir | DATA_DIR | ./ | base path where all user data is stored |
--config    | CONFIG | configs/stable-diffusion/v1-inference.yaml   				 | Path to config which constructs model. |
--ckpt 		| CKPT   | model.ckpt        				 | Path to checkpoint of Stable Diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded. |
--ckpt-dir 	| CKPT_DIR | None   				 | Path to directory with Stable Diffusion checkpoints. |
--no-download-sd-model | None | False | Don't download SD1.5 model even if no model is found. |
--do-not-download-clip | None | False | do not download CLIP model even if it's not included in the checkpoint |
--vae-dir | VAE_PATH | None  					| Path to Variational Autoencoders model | disables all settings related to VAE.
--vae-path | VAE_PATH | None | Checkpoint to use as VAE; setting this argument
--gfpgan-dir| GFPGAN_DIR | GFPGAN/			 | GFPGAN directory. |
--gfpgan-model| GFPGAN_MODEL			 | GFPGAN model file name. |
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
--embeddings-dir | EMBEDDINGS_DIR | embeddings/		 | Embeddings directory for textual inversion (default: embeddings). |
--textual-inversion-templates-dir | TEXTUAL_INVERSION_TEMPLATES_DIR | textual_inversion_templates | Directory with textual inversion templates.
--hypernetwork-dir | HYPERNETWORK_DIR | models/hypernetworks/	 | hypernetwork directory. |
--localizations-dir | LOCALIZATIONS_DIR | localizations/ | Localizations directory.
--styles-file | STYLES_FILE | styles.csv 				| Filename to use for styles. |
--ui-config-file | UI_CONFIG_FILE | 	ui-config.json	| Filename to use for UI configuration. |
--no-progressbar-hiding | None | False 			 | Do not hide progress bar in gradio UI (we hide it because it slows down ML if you have hardware acceleration in browser). |
--max-batch-count| MAX_BATCH_COUNT | 16	 | Maximum batch count value for the UI. |
--ui-settings-file | UI_SETTINGS_FILE | config.json 	| Filename to use for UI settings. |
--allow-code | None | False          				 | Allow custom script execution from web UI. |
--share | None | False               				 | Use `share=True` for gradio and make the UI accessible through their site.
--listen  | None | False            					| Launch gradio with 0.0.0.0 as server name, allowing to respond to network requests. |
--port | PORT | 7860           					| Launch gradio with given server port, you need root/admin rights for ports < 1024; defaults to 7860 if available. |
--hide-ui-dir-config | None | False  					| Hide directory configuration from web UI. |
--freeze-settings | None | False | disable editing settings |
--enable-insecure-extension-access | None | False | Enable extensions tab regardless of other options. |
--gradio-debug | None | False        					| Launch gradio with `--debug` option. |
--gradio-auth | GRADIO_AUTH | None 				| Set gradio authentication like `username:password`; or comma-delimit multiple like `u1:p1,u2:p2,u3:p3`. |
--gradio-auth-path | GRADIO_AUTH_PATH | None | Set gradio authentication file path ex. `/path/to/auth/file` same auth format as `--gradio-auth`. |
--disable-console-progressbars | None | False			| Do not output progress bars to console. |
--enable-console-prompts | None | False				| Print prompts to console when generating with txt2img and img2img. |
--api | None | False | Launch web UI with API. |
--api-auth | API_AUTH | None | Set authentication for API like `username:password`; or comma-delimit multiple like `u1:p1,u2:p2,u3:p3`. |
--api-log | None | False | Enable logging of all API requests. |
--nowebui | None | False | Only launch the API, without the UI. |
--ui-debug-mode | None | False | Don't load model to quickly launch UI. |
--device-id | DEVICE_ID | None | Select the default CUDA device to use (export `CUDA_VISIBLE_DEVICES=0,1` etc might be needed before). |
--administrator | None | False | Administrator privileges. |
--cors-allow-origins | CORS_ALLOW_ORIGINS | None | Allowed CORS origin(s) in the form of a comma-separated list (no spaces). |
--cors-allow-origins-regex | CORS_ALLOW_ORIGINS_REGEX | None | Allowed CORS origin(s) in the form of a single regular expression. |
--tls-keyfile | TLS_KEYFILE | None | Partially enables TLS, requires `--tls-certfile` to fully function. |
--tls-certfile | TLS_CERTFILE | None | Partially enables TLS, requires `--tls-keyfile` to fully function. |
--disable-tls-verify | None | False | When passed, enables the use of self-signed certificates.
--server-name | SERVER_NAME | None | Sets hostname of server. |
--no-gradio-queue | None| False | Disables gradio queue; causes the webpage to use http requests instead of websockets; was the default in earlier versions.
--gradio-allowed-path | None | None | Add path to Gradio's `allowed_paths`; make it possible to serve files from it.
--no-hashing | None | False | Disable SHA-256 hashing of checkpoints to help loading performance. |
--skip-version-check | None | False | Do not check versions of torch and xformers. |
--skip-python-version-check | None | False | Do not check versions of Python. |
--skip-torch-cuda-test | None | False | Do not check if CUDA is able to work properly. |
--skip-install | None | False | Skip installation of packages. |
--loglevel | None | None | log level; one of: CRITICAL, ERROR, WARNING, INFO, DEBUG
--log-startup | None | False | launch.py argument: print a detailed log of what's happening at startup |
--api-server-stop | None | False | enable server stop/restart/kill via api |
--timeout-keep-alive | int | 30 | set timeout_keep_alive for uvicorn |
| **PERFORMANCE** |
--xformers | None | False           					| Enable xformers for cross attention layers. |
--force-enable-xformers	| None | False				| Enable xformers for cross attention layers regardless of whether the checking code thinks you can run it; ***do not make bug reports if this fails to work***. |
--xformers-flash-attention | None | False | Enable xformers with Flash Attention to improve reproducibility (supported for SD2.x or variant only).
--opt-sdp-attention | None | False | Enable scaled dot product cross-attention layer optimization; requires PyTorch 2.*
--opt-sdp-no-mem-attention | False | None | Enable scaled dot product cross-attention layer optimization without memory efficient attention, makes image generation deterministic; requires PyTorch 2.*
--opt-split-attention | None | False 		| Force-enables Doggettx's cross-attention layer optimization. By default, it's on for CUDA-enabled systems. |
--opt-split-attention-invokeai | None | False			| Force-enables InvokeAI's cross-attention layer optimization. By default, it's on when CUDA is unavailable. |
--opt-split-attention-v1 | None | False 				| Enable older version of split attention optimization that does not consume all VRAM available. |
--opt-sub-quad-attention | None | False | Enable memory efficient sub-quadratic cross-attention layer optimization.
--sub-quad-q-chunk-size | SUB_QUAD_Q_CHUNK_SIZE | 1024 | Query chunk size for the sub-quadratic cross-attention layer optimization to use.
--sub-quad-kv-chunk-size | SUB_QUAD_KV_CHUNK_SIZE | None | KV chunk size for the sub-quadratic cross-attention layer optimization to use.
--sub-quad-chunk-threshold | SUB_QUAD_CHUNK_THRESHOLD | None | The percentage of VRAM threshold for the sub-quadratic cross-attention layer optimization to use chunking.
--opt-channelslast | None | False    					| Enable alternative layout for 4d tensors, may result in faster inference **only** on Nvidia cards with Tensor cores (16xx and higher). |
--disable-opt-split-attention | None | False 			| Force-disables cross-attention layer optimization. |
--disable-nan-check | None | False | Do not check if produced images/latent spaces have nans; useful for running without a checkpoint in CI.
--use-cpu | {all, sd, interrogate, gfpgan, bsrgan, esrgan, scunet, codeformer} | None | Use CPU as torch device for specified modules. |
--no-half     | None | False         				 | Do not switch the model to 16-bit floats. |
--precision | {full,autocast} | autocast			 | Evaluate at this precision. |
--no-half-vae | None | False         				 | Do not switch the VAE model to 16-bit floats. |
--upcast-sampling | None | False | Upcast sampling. No effect with `--no-half`. Usually produces similar results to `--no-half` with better performance while using less memory.
--medvram    | None | False          				 | Enable Stable Diffusion model optimizations for sacrificing a some performance for low VRAM usage. |
--medvram-sdxl | None | False                         | enable `--medvram` optimization just for SDXL models
--lowvram    | None | False          				 | Enable Stable Diffusion model optimizations for sacrificing a lot of speed for very low VRAM usage. |
--lowram     | None | False         				 | Load Stable Diffusion checkpoint weights to VRAM instead of RAM.
--disable-model-loading-ram-optimization | None | False | disable an optimization that reduces RAM use when loading a model |
| **FEATURES** |
--autolaunch | None | False         					| Open the web UI URL in the system's default browser upon launch. |
--theme | None | Unset         					| Open the web UI with the specified theme (`light` or `dark`). If not specified, uses the default browser theme. |
--use-textbox-seed | None | False   					| Use textbox for seeds in UI (no up/down, but possible to input long seeds). |
--disable-safe-unpickle | None | False				| Disable checking PyTorch models for malicious code. |
--ngrok | NGROK | None         				 | ngrok authtoken, alternative to gradio `--share`.
--ngrok-region | NGROK_REGION | us			 | The region in which ngrok should start.
--ngrok-options | NGROK_OPTIONS | None | The options to pass to ngrok in JSON format, e.g.: `{"authtoken_from_env":true, "basic_auth":"user:password", "oauth_provider":"google", "oauth_allow_emails":"user@asdf.com"}`
--update-check | None | None | On startup, notifies whether or not your web UI version (commit) is up-to-date with the current master branch.
--update-all-extensions | None | None | On startup, it pulls the latest updates for all extensions you have installed.
--reinstall-xformers | None | False | Force-reinstall xformers. Useful for upgrading - but remove it after upgrading or you'll reinstall xformers perpetually. |
--reinstall-torch | None | False | Force-reinstall torch. Useful for upgrading - but remove it after upgrading or you'll reinstall torch perpetually. |
--tests | TESTS | False | Run test to validate web UI functionality, see wiki topic for more details.
--no-tests | None | False | Do not run tests even if `--tests` option is specified.
--dump-sysinfo | None | False | launch.py argument: dump limited sysinfo file (without information about extensions, options) to disk and quit
--disable-all-extensions | None | False | disable all non-built-in extensions from running
--disable-extra-extensions | None | False | disable all extensions from running 
| **DEFUNCT OPTIONS** |
--show-negative-prompt | None | False 					| No longer has an effect. |
--deepdanbooru | None | False 					| No longer has an effect. |
--unload-gfpgan | None | False      				 | No longer has an effect.
--gradio-img2img-tool | GRADIO_IMG2IMG_TOOL | None | No longer has an effect. |
--gradio-inpaint-tool | GRADIO_INPAINT_TOOL | None | No longer has an effect. |
--gradio-queue | None | False | No longer has an effect. |
--add-stop-route | None | False | No longer has an effect. |
--always-batch-cond-uncond | None | False			 | No longer has an effect, move into UI under `Setting > Optimizations` |