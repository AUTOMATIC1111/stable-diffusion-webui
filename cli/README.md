# Scripts using Stable-Diffusion/Automatic API

*Note*: Start **SD/Automatic** using `python launch.py --api`
## Generate

Text-to-image with all of the possible parameters  
Supports upsampling, face restoration and grid creation  
> python generate.py --help

By default uses parameters from  `generate.json`

Parameters that are not specified will be randomized to some extent:
- Prompt will be dynamically created from template of random samples: `random.json`
- Sampler/Scheduler will be randomly picked from available ones
- CFG Scale set to 5-10

## Train

End-to-end embedding training
> python train.py --help

Combined pipeline:
1. Creates embedding  
2. Extracts images if input is movie  
3. Preprocesses images  
4. Runs training  

## Interrogate

Runs CLiP and Booru image interrogation on any provided parameters  
*(image, list of images, wildcards, folder, etc.)*
> python interrogate.py

## Promptist

Attempts to beautify the provided prompt  
> python promptist.py

## Ideas

Generate complex prompt ideas
> python ideas.py --help

## SDAPI

Utility module that handles async communication to Automatic API endpoints  
Can be used to manually execute specific commands:
> python sdapi.py progress  
> python sdapi.py interrupt

## FFMPEG

Utility module that handles video files  
Can be used to manually execute specific commands:
> ffmpeg extract --help  
> python ffmpeg.py extract --input ~/downloads/vlado.mp4 --output ./vlado --fps 2 --skipstart 3 --skipend 1

## Grid

Utility module to create image grids
> python grid.py --help

## Bench

Benchmark your Automatic
> python bench.py
