# Stable Diffusion - Automatic

*Heavily opinionated custom fork of* <https://github.com/AUTOMATIC1111/stable-diffusion-webui>  

![](wiki/images/README.jpg)

<br>

## Notes

Fork is as close as up-to-date with origin as time allows  
All code changes are merged upstream whenever possible  

Fork does differ in few things:

- Different start script  
  > ./automatic.sh  
- Drops compatibility with `python` **3.7** and requires **3.10**  
- If you're using **PyTorch 2.0** models will be auto-compiled and optimized on load  
  Using `max-tune` 
- Updated **Python** libraries to latest known compatible versions  
  e.g. `accelerate`, `transformers`, `numpy`, etc.  
- Includes opinionated **System** and **Options** configuration  
  e.g. `samplers`, `upscalers`, etc.  
- Includes reskinned **UI**  
  Black and orange dark theme with fixed width options panels and larger previews  
- Includes **SD2** configuration files  
- Ships with additional **extensions**  
  e.g. `System Info`  
- Uses simplified folder structure  
  e.g. `/train`, `/outputs/*`  
- Modified training templates  

Only Python library which is not auto-updated is `PyTorch` itself as that is very system specific  
I'm currently using **PyTorch 2.0-nightly** compiled with **CUDA 11.8** and with **Triton** optimizations:

> pip3 install --pre torch torchvision torchaudio torchtriton --extra-index-url https://download.pytorch.org/whl/nightly/cu118  
> pip show torch  
> 2.0.0.dev20230111+cu118  


<br>

## Docs

Everything is in [Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
