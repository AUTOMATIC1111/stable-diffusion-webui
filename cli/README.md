# Stable-Diffusion Productivity Scripts

*Notes*:
- Offline scripts can be used with or without **Automatic WebUI**
- Online scripts rely on **Automatic WebUI** API which should be started with `--api` parameter
- All scripts have built-in `--help` parameter that can be used to get more information

<br>

## Main Scripts

### Generate

Text-to-image with all of the possible parameters  
Supports upsampling, face restoration and grid creation  
> python generate.py

By default uses parameters from  `generate.json`

Parameters that are not specified will be randomized:
- Prompt will be dynamically created from template of random samples: `random.json`
- Sampler/Scheduler will be randomly picked from available ones
- CFG Scale set to 5-10

### Train

Textual inversion embedding training
> python train-ti.py

Combined pipeline:
1. Creates embedding  
2. Extracts images if input is movie  
3. Preprocesses images  
4. Runs training  

LoRA training
> python train-lora.py

Combined pipeline:
1. Preprocesses images  
2. Runs training  

[Detailed documentation](https://github.com/vladmandic/automatic/wiki/Process.md)

LoRA extract from model
> python moidules/lora-extract.py

<br>

## Auxiliary Scripts

### Benchmark

Benchmark your **Automatic WebUI**  
Note: Requires SD API  

> python modules/bench.py

### Embedding Previews

Create previews of embeddings using preview templates  
Note: Requires SD API  

> python modules/preview-embeddings.py

## Grid

Create flexible image grids from any number of images  
Note: Offline tool  

> python modiles/grid.py

### Image Watermark

Create invisible image watermark and remove existing EXIF tags  
Note: Offline tool  

> python modules/image-watermark.py

### Interrogate

Runs CLiP and Booru image interrogation  
Note: Requires SD API  

> python modules/interrogate.py

### Interrogate-Offline

Standalone implementation of GiT, CLiP and ViT image interrogation  
Note: Offline tool  

> python modules/interrogate-offline.py

### Models Previews

Create previews of models using built-in templates  
Note: Requires SD API  

> python modules/preview-models.py

### Palette Extract

Extract color palette from image(s)  
Note: Offline tool  

> python modules/palette-extract.py

### Image Process

Run image processing to extract face/body segments and run resolution/blur/dynamic-range checks  
Note: Offline except for interrogate to generate caption files which requires SD API  

> python modules/process.py

[Detailed documentation](https://github.com/vladmandic/automatic/wiki/Process.md)

### Prompt Ideas

Generate complex prompt ideas
Note: Offline tool  

> python modules/prompt-ideas.py

### Prompt Promptist

Attempts to beautify the provided prompt  
Note: Offline tool  

> python modules/promptist.py

### Training Loss-Chart

Create loss-chart from training log
Note: Offline tool, may require adjustment to train paths if used with other repos  

> python modules/train-losschart.py

### Training Loss-Rate

Create customizable loss rate to be used in training
Note: Offline tool  

> python modules/train-lossrate.py

### Video Extract

Extract frames from video files  
Note: Offline tool  

> python modules/video-extract.py

<br>

## Utility Scripts
### SDAPI

Utility module that handles async communication to Automatic API endpoints  
Note: Requires SD API  

Can be used to manually execute specific commands:
> python sdapi.py progress  
> python sdapi.py interrupt
> python sdapi.py shutdown
