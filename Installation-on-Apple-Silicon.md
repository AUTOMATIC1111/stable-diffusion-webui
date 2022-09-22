**IMPORTANT** While the web UI runs fine, there are still certain issues when running this fork on Apple Silicon.
The only 2 samplers that work (at the time of writing this) are `Euler` and `DPM2` - all others result in a black screen.
Upscaling works, but only using the real-ESRGAN models.

First get the weights checkpoint download started - it's big:

Sign up at https://huggingface.co
Go to the Stable diffusion diffusion model page
Accept the terms and click Access Repository:
Download sd-v1-4.ckpt (4.27 GB) and note where you have saved it (probably the Downloads folder)

1. `brew install cmake protobuf rust`
2. `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o Miniconda3-latest-MacOSX-arm64.sh`
3. `/bin/bash Miniconda3-latest-MacOSX-arm64.sh`
4. `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui`
5. `cd stable-diffusion-webui`
6. Run the following commands one by one: 
```
git clone https://github.com/CompVis/stable-diffusion.git repositories/stable-diffusion
 
git clone https://github.com/CompVis/taming-transformers.git repositories/taming-transformers

git clone https://github.com/sczhou/CodeFormer.git repositories/CodeFormer
    
git clone https://github.com/salesforce/BLIP.git repositories/BLIP
```

7. `conda create --name web_ui python=3.10`
8. `conda activate web_ui`
9. `pip install -r requirements.txt`
10. `conda install pytorch torchvision torchaudio -c pytorch-nightly`
11. At this point, move the downloaded `sd-v1-4.ckpt` file into `stable-diffusion-webui/models/`. You will know it's the right folder since there's a text file named `Put Stable Diffusion checkpoints here.txt` in it.
12. `conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1`
13. `python webui.py --precision full --no-half --opt-split-attention-v1`

It is possible that after running `webui.py` you get error messages saying certain packages are missing. Install the missing package, go back to step 13 and repeat.

#### Common Errors

##### Error

`ImportError: dlopen(.venv/lib/python3.10/site-packages/google/protobuf/pyext/_message.cpython-310-darwin.so, 0x0002): symbol not found in flat namespace (__ZN6google8protobuf15FieldDescriptor12TypeOnceInitEPKS1_)`

##### Solution

Downgrade Protobuf using `pip install protobuf==3.19.4`